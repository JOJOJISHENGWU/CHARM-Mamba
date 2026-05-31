# Efficiency Reporting Notes

This file documents how to interpret efficiency numbers in a paper-aligned manner.

## Reported dimensions

- Trainable parameters
- FLOPs
- Inference latency
- Training latency
- GPU memory usage

## Measurement principles

- Use the same input shape and horizon settings when comparing methods.
- Use the same hardware/software environment per comparison group.
- Distinguish clearly between:
  - structural compute cost (e.g., FLOPs/inference), and
  - optimization/adaptation cost (trainable parameters, training latency, memory).

## Paper alignment

The paper claims that HEA reduces trainable parameters and adaptation cost while keeping backbone inference structure unchanged. Any reproduced table should preserve this interpretation and measurement protocol.
