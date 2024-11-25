# Dilithium-Side-channel-Attack-and-leakage-analysis
This work primarily includes:

- Template attacks on Dilithium  
- Approximate universal DPA on AES  
- Commonly used side-channel leakage analysis tools 

The template attack on Dilithium is based on the principles described in:  
*Profiling Side-Channel Attacks on Dilithium: A Small Bit-Fiddling Leak Breaks It All*,  
[URL](https://eprint.iacr.org/2022/106).

The approximate universal DPA outlines a non-profiling attack method applied in scenarios where only a single leakage point of the AES first-round SBox is available.

The side-channel leakage analysis tools primarily include functionalities such as:  
- Concurrent first-order t-tests  
- Mutual information computation  
- Signal-to-noise ratio (SNR) calculation of traces  
