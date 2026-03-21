"""
Six methods for sample-based quantum diagonalization.

Classical (NQS-based):
  - NQS+SQD:      Train NQS → sample → SQD diagonalize (two-stage)
  - NQS+SKQD:     Train NQS → sample → SKQD Krylov diagonalize (two-stage)
  - HI+NQS+SQD:   NQS and SQD in iterative feedback loop (self-consistent)

Quantum (circuit-based):
  - QC+SQD:       Quantum circuit → sample → SQD diagonalize
  - QC+SKQD:      Quantum circuit Trotter evolution → sample → SKQD
  - HI-VQE:       Quantum circuit and SQD in iterative loop
"""
