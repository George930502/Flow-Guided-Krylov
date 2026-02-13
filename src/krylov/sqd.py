"""
Sample-based Quantum Diagonalization (SQD).

Implements the SQD algorithm from:
    "Chemistry Beyond the Scale of Exact Diagonalization on a Quantum-Centric Supercomputer"

In this pipeline, the NF-NQS replaces the quantum circuit as the sampler.
The algorithm:
1. Takes NF-NQS sampled configurations as input
2. Filters for correct particle number (NF already conserves particles)
3. Creates K batches with spin symmetry enhancement
4. Diagonalizes projected Hamiltonian in each batch independently
5. Self-consistently updates orbital occupancies and re-batches
6. Performs energy-variance extrapolation across batches
"""

import torch
import numpy as np
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass

# Support both package imports and direct script execution
try:
    from ..utils.gpu_linalg import gpu_eigh
except ImportError:
    try:
        from utils.gpu_linalg import gpu_eigh
    except ImportError:
        gpu_eigh = None


@dataclass
class SQDConfig:
    """Configuration for SQD solver."""

    # Batch parameters
    num_batches: int = 5              # K: number of independent batches
    batch_size: int = 0               # d: configs per batch (0 = auto)

    # Self-consistent configuration recovery
    self_consistent_iters: int = 3    # Max iterations for self-consistency
    occupancy_convergence: float = 0.01  # Convergence threshold for orbital occupancies

    # Spin symmetry
    spin_penalty: float = 0.0         # Lambda for S^2 penalty (0 = disabled)
    use_spin_symmetry_enhancement: bool = True  # Spin-up/down recombination

    # Configuration recovery (for non-particle-conserving samplers)
    enable_config_recovery: bool = False  # Disabled by default (NF conserves particles)
    recovery_delta: float = 0.01      # Modified ReLU parameter delta
    recovery_h: float = 0.0           # Modified ReLU corner (0 = auto from filling)


class SQDSolver:
    """
    Sample-based Quantum Diagonalization (SQD).

    Replaces quantum circuit sampling with NF-NQS sampling.
    Implements batch diagonalization with self-consistent orbital occupancies
    from the IBM quantum-centric supercomputer paper.
    """

    def __init__(self, hamiltonian, config: Optional[SQDConfig] = None):
        self.hamiltonian = hamiltonian
        self.config = config or SQDConfig()
        self.num_sites = hamiltonian.num_sites

        # Molecular properties
        self._is_molecular = hasattr(hamiltonian, 'n_alpha')
        if self._is_molecular:
            self.n_alpha = hamiltonian.n_alpha
            self.n_beta = hamiltonian.n_beta
            self.n_orbitals = hamiltonian.n_orbitals
            self.n_electrons = self.n_alpha + self.n_beta
        else:
            self.n_alpha = None
            self.n_beta = None
            self.n_orbitals = self.num_sites // 2
            self.n_electrons = None

    def run(self, nf_basis: torch.Tensor, progress: bool = True) -> Dict[str, Any]:
        """
        Run SQD algorithm on NF-NQS sampled configurations.

        Args:
            nf_basis: Tensor of configurations (n_configs, num_sites)
            progress: Whether to print progress

        Returns:
            Dictionary with energy, batch results, and diagnostics
        """
        cfg = self.config
        device = nf_basis.device
        n_configs = len(nf_basis)

        print(f"SQD: {n_configs} input configurations, {cfg.num_batches} batches")

        # Step 1: Filter for correct particle number
        valid_configs, invalid_configs = self._filter_particle_number(nf_basis)
        print(f"  Valid configs (correct N): {len(valid_configs)}")
        if len(invalid_configs) > 0:
            print(f"  Invalid configs (wrong N): {len(invalid_configs)}")

        # Determine batch size
        batch_size = cfg.batch_size
        if batch_size <= 0:
            # Auto: use all valid configs divided across batches
            batch_size = len(valid_configs)
        print(f"  Batch size (d): {batch_size}")

        # Step 2: Initialize orbital occupancies from HF state
        if self._is_molecular:
            hf_state = self.hamiltonian.get_hf_state().float()
            orbital_occ = hf_state.cpu().numpy()
        else:
            orbital_occ = np.full(self.num_sites, 0.5)

        # Step 3: Self-consistent loop
        all_configs = valid_configs
        best_energy = float('inf')
        best_results = None

        for sc_iter in range(max(1, cfg.self_consistent_iters)):
            if progress:
                print(f"\n  Self-consistent iteration {sc_iter + 1}/{cfg.self_consistent_iters}")

            # Configuration recovery (if enabled and there are invalid configs)
            if cfg.enable_config_recovery and len(invalid_configs) > 0:
                recovered = self._recover_configurations(
                    invalid_configs, orbital_occ, device
                )
                if len(recovered) > 0:
                    all_configs = torch.cat([valid_configs, recovered], dim=0)
                    all_configs = torch.unique(all_configs, dim=0)
                    if progress:
                        print(f"    Recovered {len(recovered)} configs -> {len(all_configs)} total")
            else:
                all_configs = valid_configs

            # Create K batches
            batches = self._create_batches(all_configs, batch_size, cfg.num_batches)

            # Diagonalize each batch
            batch_results = []
            for k, batch in enumerate(batches):
                result = self._diagonalize_batch(batch, k)
                batch_results.append(result)
                if progress:
                    print(f"    Batch {k+1}: E = {result['energy']:.8f} Ha "
                          f"({len(batch)} configs, var = {result['variance']:.2e})")

            # Compute orbital occupancies from eigenstates
            new_occ = self._compute_orbital_occupancies(batch_results)

            # Check convergence
            occ_change = np.max(np.abs(new_occ - orbital_occ))
            orbital_occ = new_occ
            if progress:
                print(f"    Max occupancy change: {occ_change:.6f}")

            # Track best energy
            energies = [r['energy'] for r in batch_results]
            mean_energy = np.mean(energies)
            if mean_energy < best_energy:
                best_energy = mean_energy
                best_results = batch_results

            if occ_change < cfg.occupancy_convergence and sc_iter > 0:
                if progress:
                    print(f"    Converged after {sc_iter + 1} iterations")
                break

        # Step 4: Energy-variance extrapolation
        extrapolated_energy, ev_results = self._energy_variance_extrapolation(best_results)

        # Pick best energy: minimum of batch energies and extrapolated
        batch_energies = [r['energy'] for r in best_results]
        min_batch_energy = min(batch_energies)

        # Use minimum batch energy (variational upper bound)
        final_energy = min_batch_energy
        if extrapolated_energy is not None and extrapolated_energy < final_energy:
            # Extrapolation can go below variational bound
            final_energy = extrapolated_energy

        print(f"\n  SQD Results:")
        print(f"    Min batch energy:    {min_batch_energy:.8f} Ha")
        print(f"    Mean batch energy:   {np.mean(batch_energies):.8f} Ha")
        print(f"    Std batch energy:    {np.std(batch_energies):.8f} Ha")
        if extrapolated_energy is not None:
            print(f"    Extrapolated energy: {extrapolated_energy:.8f} Ha")
        print(f"    Final energy:        {final_energy:.8f} Ha")

        return {
            "energy": final_energy,
            "min_batch_energy": min_batch_energy,
            "mean_batch_energy": float(np.mean(batch_energies)),
            "std_batch_energy": float(np.std(batch_energies)),
            "extrapolated_energy": extrapolated_energy,
            "batch_energies": batch_energies,
            "batch_variances": [r['variance'] for r in best_results],
            "batch_sizes": [r['batch_size'] for r in best_results],
            "energy_variance_fit": ev_results,
            "self_consistent_iters": sc_iter + 1,
            "orbital_occupancies": orbital_occ.tolist(),
            "num_input_configs": n_configs,
            "num_valid_configs": len(valid_configs),
        }

    def _filter_particle_number(
        self, configs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split configs into correct and incorrect particle number sets."""
        if not self._is_molecular:
            return configs, torch.empty(0, self.num_sites, device=configs.device)

        n_orb = self.n_orbitals
        alpha_counts = configs[:, :n_orb].sum(dim=1)
        beta_counts = configs[:, n_orb:].sum(dim=1)

        valid_mask = (alpha_counts == self.n_alpha) & (beta_counts == self.n_beta)

        return configs[valid_mask], configs[~valid_mask]

    def _recover_configurations(
        self,
        wrong_configs: torch.Tensor,
        orbital_occ: np.ndarray,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Recover configurations with wrong particle number by flipping orbitals.

        Uses the modified ReLU weighting from the paper:
        w(y) = delta * y / h          if y <= h
        w(y) = delta + (1-delta)(y-h)/(1-h)  if y > h

        where y = |x_ps - n_ps|, h = filling factor, delta = 0.01
        """
        if len(wrong_configs) == 0:
            return torch.empty(0, self.num_sites, device=device)

        cfg = self.config
        n_orb = self.n_orbitals
        delta = cfg.recovery_delta
        h = cfg.recovery_h if cfg.recovery_h > 0 else self.n_electrons / self.num_sites

        recovered = []
        occ_tensor = torch.tensor(orbital_occ, dtype=torch.float32, device=device)

        for config in wrong_configs:
            new_config = config.clone()

            # Fix alpha electrons
            alpha_part = new_config[:n_orb]
            alpha_count = int(alpha_part.sum().item())
            alpha_occ = occ_tensor[:n_orb]

            if alpha_count != self.n_alpha:
                new_config[:n_orb] = self._fix_spin_sector(
                    alpha_part, alpha_occ, self.n_alpha, alpha_count, delta, h
                )

            # Fix beta electrons
            beta_part = new_config[n_orb:]
            beta_count = int(beta_part.sum().item())
            beta_occ = occ_tensor[n_orb:]

            if beta_count != self.n_beta:
                new_config[n_orb:] = self._fix_spin_sector(
                    beta_part, beta_occ, self.n_beta, beta_count, delta, h
                )

            recovered.append(new_config)

        if recovered:
            result = torch.stack(recovered)
            return torch.unique(result, dim=0)
        return torch.empty(0, self.num_sites, device=device)

    def _fix_spin_sector(
        self,
        sector: torch.Tensor,
        occ: torch.Tensor,
        target_n: int,
        current_n: int,
        delta: float,
        h: float,
    ) -> torch.Tensor:
        """Fix particle number in a spin sector by probabilistic bit flipping."""
        result = sector.clone()
        diff = current_n - target_n

        if diff > 0:
            # Too many particles: flip some occupied -> unoccupied
            occupied = (result == 1).nonzero(as_tuple=True)[0]
            if len(occupied) == 0:
                return result
            # Weight: high weight for orbitals that should be empty
            distances = torch.abs(1.0 - occ[occupied])
            weights = self._modified_relu(distances, delta, h)
            weights = weights / (weights.sum() + 1e-12)
            n_flip = min(abs(diff), len(occupied))
            indices = torch.multinomial(weights, n_flip, replacement=False)
            result[occupied[indices]] = 0

        elif diff < 0:
            # Too few particles: flip some unoccupied -> occupied
            empty = (result == 0).nonzero(as_tuple=True)[0]
            if len(empty) == 0:
                return result
            # Weight: high weight for orbitals that should be occupied
            distances = torch.abs(0.0 - occ[empty])
            weights = self._modified_relu(distances, delta, h)
            weights = weights / (weights.sum() + 1e-12)
            n_flip = min(abs(diff), len(empty))
            indices = torch.multinomial(weights, n_flip, replacement=False)
            result[empty[indices]] = 1

        return result

    @staticmethod
    def _modified_relu(y: torch.Tensor, delta: float, h: float) -> torch.Tensor:
        """Modified ReLU function from the paper (Eq. in supplement)."""
        result = torch.zeros_like(y)
        low_mask = y <= h
        high_mask = ~low_mask

        if h > 0:
            result[low_mask] = delta * y[low_mask] / h
        result[high_mask] = delta + (1.0 - delta) * (y[high_mask] - h) / max(1.0 - h, 1e-12)

        return result.clamp(min=1e-12)

    def _create_batches(
        self,
        configs: torch.Tensor,
        batch_size: int,
        num_batches: int,
    ) -> List[torch.Tensor]:
        """
        Create K batches of configurations.

        If spin symmetry enhancement is enabled, samples sqrt(d/2) configs,
        extracts unique spin-up/down parts, and forms all combinations.
        This facilitates singlet state construction.
        """
        cfg = self.config
        n_configs = len(configs)

        if n_configs == 0:
            return [configs for _ in range(num_batches)]

        batches = []

        for k in range(num_batches):
            if cfg.use_spin_symmetry_enhancement and self._is_molecular:
                batch = self._create_spin_enhanced_batch(configs, batch_size, k)
            else:
                # Simple random subsampling
                if n_configs <= batch_size:
                    batch = configs.clone()
                else:
                    gen = torch.Generator(device='cpu')
                    gen.manual_seed(42 + k)
                    perm = torch.randperm(n_configs, generator=gen)[:batch_size]
                    batch = configs[perm]

            batches.append(batch)

        return batches

    def _create_spin_enhanced_batch(
        self,
        configs: torch.Tensor,
        batch_size: int,
        batch_index: int,
    ) -> torch.Tensor:
        """
        Create a batch with spin symmetry enhancement.

        Sample sqrt(d/2) configs, extract unique spin-up and spin-down parts,
        form all alpha-beta combinations to facilitate singlet state construction.
        """
        n_orb = self.n_orbitals
        n_configs = len(configs)

        # Sample sqrt(d/2) configurations
        n_sample = max(1, int(np.sqrt(batch_size / 2)))
        n_sample = min(n_sample, n_configs)

        gen = torch.Generator(device='cpu')
        gen.manual_seed(42 + batch_index)
        perm = torch.randperm(n_configs, generator=gen)[:n_sample]
        sampled = configs[perm]

        # Extract unique alpha and beta parts
        alpha_parts = sampled[:, :n_orb]
        beta_parts = sampled[:, n_orb:]

        unique_alpha = torch.unique(alpha_parts, dim=0)
        unique_beta = torch.unique(beta_parts, dim=0)

        # Form all alpha-beta combinations
        n_alpha_unique = len(unique_alpha)
        n_beta_unique = len(unique_beta)

        # If too many combinations, subsample
        max_combos = batch_size
        total_combos = n_alpha_unique * n_beta_unique

        if total_combos <= max_combos:
            # All combinations
            alpha_expanded = unique_alpha.repeat_interleave(n_beta_unique, dim=0)
            beta_expanded = unique_beta.repeat(n_alpha_unique, 1)
            batch = torch.cat([alpha_expanded, beta_expanded], dim=1)
        else:
            # Random subset of combinations
            combos = []
            gen2 = torch.Generator(device='cpu')
            gen2.manual_seed(137 + batch_index)
            for _ in range(max_combos):
                ai = torch.randint(n_alpha_unique, (1,), generator=gen2).item()
                bi = torch.randint(n_beta_unique, (1,), generator=gen2).item()
                combo = torch.cat([unique_alpha[ai], unique_beta[bi]])
                combos.append(combo)
            batch = torch.stack(combos)

        # Filter for correct particle number
        if self._is_molecular:
            alpha_counts = batch[:, :n_orb].sum(dim=1)
            beta_counts = batch[:, n_orb:].sum(dim=1)
            valid = (alpha_counts == self.n_alpha) & (beta_counts == self.n_beta)
            batch = batch[valid]

        batch = torch.unique(batch, dim=0)
        return batch

    def _diagonalize_batch(
        self, batch: torch.Tensor, batch_index: int
    ) -> Dict[str, Any]:
        """
        Project Hamiltonian into batch subspace and diagonalize.

        Optionally adds S^2 penalty for spin contamination control.
        """
        n = len(batch)
        if n == 0:
            return {
                'energy': float('inf'),
                'variance': float('inf'),
                'eigenvector': None,
                'batch_size': 0,
                'batch_index': batch_index,
            }

        # Build projected Hamiltonian
        H_matrix = self.hamiltonian.matrix_elements(batch, batch)
        H_np = H_matrix.detach().cpu().numpy().real.astype(np.float64)

        # Symmetrize
        H_np = 0.5 * (H_np + H_np.T)

        # Add S^2 penalty if configured
        if self.config.spin_penalty > 0 and self._is_molecular:
            S2_matrix = self._compute_s2_matrix(batch)
            # Target: singlet (s=0), so s(s+1) = 0
            # Penalty: lambda * (S^2 - 0)^2 = lambda * S^4
            # Approximate: lambda * S^2 is simpler and often sufficient
            H_np = H_np + self.config.spin_penalty * (S2_matrix @ S2_matrix)

        # Diagonalize
        eigenvalues, eigenvectors = np.linalg.eigh(H_np)
        energy = float(eigenvalues[0])
        ground_state = eigenvectors[:, 0]

        # Compute variance: <H^2> - <H>^2
        H2_expectation = float(ground_state @ H_np @ H_np @ ground_state)
        variance = H2_expectation - energy ** 2
        variance = max(0.0, variance)  # Clamp negative due to numerics

        return {
            'energy': energy,
            'variance': variance,
            'eigenvector': ground_state,
            'batch': batch,
            'batch_size': n,
            'batch_index': batch_index,
        }

    def _compute_s2_matrix(self, configs: torch.Tensor) -> np.ndarray:
        """
        Compute S^2 matrix elements in the configuration basis.

        S^2 = S_z^2 + S_z + S_- S_+

        For computational basis states, S_z is diagonal.
        S_+S_- connects configs differing by one alpha->beta or beta->alpha flip.
        """
        n = len(configs)
        n_orb = self.n_orbitals
        S2 = np.zeros((n, n), dtype=np.float64)

        configs_np = configs.cpu().numpy()

        for i in range(n):
            alpha_i = configs_np[i, :n_orb]
            beta_i = configs_np[i, n_orb:]

            # Diagonal: S_z^2 + S_z term
            ms_i = 0.5 * (np.sum(alpha_i) - np.sum(beta_i))
            # <i|S^2|i> = S_z(S_z + 1) + number of same-spin pairs
            # More precisely: Sz^2 + Sz + sum_p n_{p,alpha}(1-n_{p,beta})
            S2[i, i] = ms_i * (ms_i + 1.0)
            for p in range(n_orb):
                S2[i, i] += alpha_i[p] * (1 - beta_i[p])

            # Off-diagonal: S_- S_+ connections
            for j in range(i + 1, n):
                alpha_j = configs_np[j, :n_orb]
                beta_j = configs_np[j, n_orb:]

                # S_+ flips beta->alpha at site p, S_- flips alpha->beta at site q
                # Check if configs differ by exactly one alpha-beta swap
                alpha_diff = alpha_i - alpha_j
                beta_diff = beta_i - beta_j

                # Should have exactly one +1 and one -1 in alpha, opposite in beta
                alpha_plus = np.where(alpha_diff == 1)[0]
                alpha_minus = np.where(alpha_diff == -1)[0]
                beta_plus = np.where(beta_diff == 1)[0]
                beta_minus = np.where(beta_diff == -1)[0]

                if (len(alpha_plus) == 1 and len(alpha_minus) == 1 and
                    len(beta_plus) == 1 and len(beta_minus) == 1 and
                    alpha_plus[0] == beta_minus[0] and alpha_minus[0] == beta_plus[0]):
                    # This is an S+S- connection with magnitude -1
                    S2[i, j] = -1.0
                    S2[j, i] = -1.0

        return S2

    def _compute_orbital_occupancies(
        self, batch_results: List[Dict[str, Any]]
    ) -> np.ndarray:
        """
        Compute average orbital occupancies from batch eigenstates.

        n_{p,sigma} = (1/K) sum_k <psi^(k)| n_hat_{p,sigma} |psi^(k)>
        """
        occupancies = np.zeros(self.num_sites, dtype=np.float64)
        n_valid = 0

        for result in batch_results:
            if result['eigenvector'] is None or result['batch'] is None:
                continue

            coeffs = result['eigenvector']  # (d,)
            configs = result['batch'].cpu().numpy().astype(np.float64)  # (d, num_sites)

            # <psi|n_hat_p|psi> = sum_{i,j} c_i* c_j <i|n_p|j> = sum_i |c_i|^2 * x_i_p
            probs = coeffs ** 2  # (d,)
            occ_k = probs @ configs  # (num_sites,)

            occupancies += occ_k
            n_valid += 1

        if n_valid > 0:
            occupancies /= n_valid

        return occupancies

    def _energy_variance_extrapolation(
        self, batch_results: List[Dict[str, Any]]
    ) -> Tuple[Optional[float], Dict[str, Any]]:
        """
        Energy-variance extrapolation across batches.

        Linear fit: delta_E ~ a * (Delta_H / E^2)
        Extrapolate to Delta_H = 0 to estimate true ground state energy.
        """
        energies = []
        variances = []

        for r in batch_results:
            if r['energy'] != float('inf') and r['variance'] != float('inf'):
                energies.append(r['energy'])
                variances.append(r['variance'])

        if len(energies) < 3:
            return None, {"fit_quality": "insufficient_data", "n_points": len(energies)}

        energies = np.array(energies)
        variances = np.array(variances)

        # Compute Delta_H / E^2
        x = variances / (energies ** 2)
        y = energies

        # Linear fit: E = E_T + a * (Delta_H / E^2)
        # Use least squares
        A = np.vstack([x, np.ones(len(x))]).T
        try:
            result = np.linalg.lstsq(A, y, rcond=None)
            slope, intercept = result[0]
            residuals = result[1] if len(result[1]) > 0 else None

            # R^2 quality metric
            ss_res = np.sum((y - (slope * x + intercept)) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1.0 - ss_res / max(ss_tot, 1e-30)

            extrapolated = float(intercept)

            return extrapolated, {
                "extrapolated_energy": extrapolated,
                "slope": float(slope),
                "r_squared": float(r_squared),
                "n_points": len(energies),
                "fit_quality": "good" if r_squared > 0.8 else "poor",
            }

        except np.linalg.LinAlgError:
            return None, {"fit_quality": "fit_failed", "n_points": len(energies)}
