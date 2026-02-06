# AWS EC2 Instance Selection Guide for PFFDTD Simulation

## Key Requirements Analysis

### 1. **Memory Requirements (CRITICAL)**
- **Primary constraint**: `u_out` array = `Nr × Nt × 8 bytes` (double precision)
- **Current example**: 
  - Nr = 4,755,606 receivers (with 0.2m spacing)
  - Nt ≈ 3,856 time steps
  - **Required**: ~146 GB just for `u_out` array
- **Additional memory needed**:
  - GPU memory: ~1-2 GB per GPU
  - Host buffers: ~50-100 GB for intermediate arrays
  - **Total recommended**: 200-250 GB RAM minimum

### 2. **GPU Requirements**
- **CUDA support**: Requires NVIDIA GPUs
- **Multi-GPU support**: Code automatically uses all available GPUs via `CUDA_VISIBLE_DEVICES`
- **GPU memory**: ~8 GB VRAM per GPU is sufficient for most simulations
- **Compute capability**: Code compiles for `sm_89` (Ampere/Turing), but can be adjusted

### 3. **Storage Requirements**
- **Input files**: ~100-500 MB (HDF5 files)
- **Output files**: 
  - `sim_outs.h5`: ~Nr × Nt × 4 bytes (single precision) = ~73 GB for current example
  - VTKHDF output: Similar size
  - **Total storage needed**: 100-200 GB recommended

### 4. **Compute Requirements**
- **CPU**: Moderate (mainly for I/O and coordination)
- **Network**: Fast local storage (EBS or instance store) recommended
- **Duration**: Long-running (hours to days depending on simulation size)

---

## Cost-Optimized AWS EC2 Recommendations

### **Option 1: Memory-Optimized GPU Instances (Best for Large Simulations)**

#### **g5.12xlarge** (Recommended for your current workload)
- **Specs**: 
  - 4× NVIDIA A10G GPUs (24 GB VRAM each)
  - 192 GB RAM
  - 48 vCPUs
  - 3.8 TB NVMe SSD
- **On-demand**: ~$5.67/hour (~$136/day)
- **Spot**: ~$1.70/hour (~$41/day) - **70% savings**
- **Pros**: 
  - Excellent memory/GPU ratio
  - Multi-GPU support (4 GPUs)
  - Fast local NVMe storage
- **Cons**: May need to reduce receiver count slightly

#### **g5.16xlarge** (If you need more memory)
- **Specs**:
  - 1× NVIDIA A10G GPU (24 GB VRAM)
  - 256 GB RAM
  - 64 vCPUs
  - 3.8 TB NVMe SSD
- **On-demand**: ~$1.21/hour (~$29/day)
- **Spot**: ~$0.36/hour (~$8.64/day)
- **Pros**: More RAM, single GPU (simpler)
- **Cons**: Only 1 GPU (no multi-GPU scaling)

#### **g5.24xlarge** (Maximum memory)
- **Specs**:
  - 4× NVIDIA A10G GPUs (24 GB VRAM each)
  - 384 GB RAM
  - 96 vCPUs
  - 3.8 TB NVMe SSD
- **On-demand**: ~$10.14/hour (~$243/day)
- **Spot**: ~$3.04/hour (~$73/day)
- **Pros**: Plenty of RAM for very large simulations
- **Cons**: Expensive

### **Option 2: Reduce Memory Requirements (More Cost-Effective)**

**Strategy**: Increase `receiver_grid_spacing` to reduce `Nr`

| Spacing | Approx. Nr | u_out Size | Min RAM Needed | Recommended Instance |
|---------|------------|------------|----------------|---------------------|
| 0.05m   | ~4.7M      | ~146 GB    | 200 GB         | g5.24xlarge         |
| 0.1m    | ~600K      | ~18 GB     | 50 GB          | g5.2xlarge          |
| 0.2m    | ~75K       | ~2.3 GB    | 16 GB          | g5.xlarge           |
| 0.5m    | ~12K       | ~0.4 GB    | 16 GB          | g5.xlarge           |
| 1.0m    | ~1.5K      | ~0.05 GB   | 16 GB          | g5.xlarge           |

#### **g5.2xlarge** (Good balance with 0.1m spacing)
- **Specs**:
  - 1× NVIDIA A10G GPU (24 GB VRAM)
  - 32 GB RAM
  - 8 vCPUs
  - 1.9 TB NVMe SSD
- **On-demand**: ~$1.01/hour (~$24/day)
- **Spot**: ~$0.30/hour (~$7.20/day)
- **Best for**: Simulations with 0.1-0.2m receiver spacing

#### **g5.xlarge** (Most cost-effective for smaller simulations)
- **Specs**:
  - 1× NVIDIA A10G GPU (24 GB VRAM)
  - 16 GB RAM
  - 4 vCPUs
  - 1.9 TB NVMe SSD
- **On-demand**: ~$0.51/hour (~$12/day)
- **Spot**: ~$0.15/hour (~$3.60/day)
- **Best for**: Simulations with 0.2m+ receiver spacing

### **Option 3: Alternative GPU Instances (Older/Cheaper)**

#### **g4dn.xlarge** (Budget option)
- **Specs**:
  - 1× NVIDIA T4 GPU (16 GB VRAM)
  - 16 GB RAM
  - 4 vCPUs
  - 125 GB NVMe SSD
- **On-demand**: ~$0.526/hour (~$12.62/day)
- **Spot**: ~$0.16/hour (~$3.84/day)
- **Pros**: Cheaper, still good performance
- **Cons**: Less VRAM, older architecture

#### **g4dn.2xlarge**
- **Specs**:
  - 1× NVIDIA T4 GPU (16 GB VRAM)
  - 32 GB RAM
  - 8 vCPUs
  - 225 GB NVMe SSD
- **On-demand**: ~$0.752/hour (~$18/day)
- **Spot**: ~$0.23/hour (~$5.52/day)

---

## Cost Optimization Strategies

### 1. **Use Spot Instances** (70-90% savings)
- **Risk**: Can be interrupted (2-minute warning)
- **Mitigation**: 
  - Save checkpoints frequently
  - Use Spot Fleet with multiple instance types
  - Consider Spot Blocks for guaranteed 1-6 hours
- **Best for**: Non-time-critical simulations

### 2. **Right-Size Your Simulation**
- **Most impactful**: Increase `receiver_grid_spacing`
  - 0.2m → 0.5m: 16× fewer receivers, 16× less memory
  - Often sufficient for visualization purposes
- **Trade-off**: Lower spatial resolution in visualization

### 3. **Use Reserved Instances** (If running frequently)
- **1-year Reserved**: ~40% discount
- **3-year Reserved**: ~60% discount
- **Best for**: Regular, predictable workloads

### 4. **Optimize Storage**
- **Use instance store** (NVMe) for temporary files
- **EBS gp3** (not gp2) for persistent storage - 20% cheaper
- **Delete intermediate files** after processing
- **Compress outputs** (already using GZIP level 3)

### 5. **Multi-GPU Efficiency**
- Code automatically uses all GPUs
- **g5.12xlarge** (4 GPUs) provides ~4× speedup
- **Cost per GPU-hour**: Often cheaper than single-GPU instances

---

## Recommended Workflow

### **Step 1: Test Locally/Small Instance**
1. Start with **g5.xlarge** (Spot: ~$0.15/hour)
2. Use `receiver_grid_spacing=0.5m` or `1.0m`
3. Verify simulation completes successfully
4. Check actual memory usage

### **Step 2: Scale Up Based on Results**
- If memory is the bottleneck → Use larger instance (g5.2xlarge, g5.12xlarge)
- If compute is the bottleneck → Use multi-GPU instance (g5.12xlarge)
- If both → Optimize simulation parameters first

### **Step 3: Production Run**
- Use **Spot instances** for cost savings
- Monitor with CloudWatch
- Set up auto-shutdown after completion
- Use Spot Fleet for availability

---

## Memory Calculation Formula

```python
# Calculate required memory
Nr = number_of_receivers  # Depends on receiver_grid_spacing
Nt = int(duration / Ts)   # Duration in seconds, Ts from sim_consts
u_out_size_gb = (Nr * Nt * 8) / (1024**3)  # 8 bytes per double

# Add overhead
total_ram_needed_gb = u_out_size_gb * 1.5  # 50% overhead for other arrays

# Recommended instance RAM should be >= total_ram_needed_gb
```

---

## Quick Reference: Instance Comparison

| Instance | GPUs | GPU VRAM | RAM | NVMe | On-Demand/hr | Spot/hr | Best For |
|----------|------|----------|-----|------|--------------|---------|----------|
| g5.xlarge | 1 | 24 GB | 16 GB | 1.9 TB | $0.51 | $0.15 | Small sims (0.2m+) |
| g5.2xlarge | 1 | 24 GB | 32 GB | 1.9 TB | $1.01 | $0.30 | Medium sims (0.1m) |
| g5.12xlarge | 4 | 96 GB | 192 GB | 3.8 TB | $5.67 | $1.70 | Large sims (0.05m) |
| g5.24xlarge | 4 | 96 GB | 384 GB | 3.8 TB | $10.14 | $3.04 | Very large sims |
| g4dn.xlarge | 1 | 16 GB | 16 GB | 125 GB | $0.526 | $0.16 | Budget option |

---

## Additional Tips

1. **Region Selection**: Choose regions with lower Spot prices (us-east-1, us-west-2 often cheapest)
2. **Auto-Shutdown**: Use AWS Systems Manager or Lambda to auto-stop instances
3. **Monitoring**: Set up CloudWatch alarms for memory/GPU utilization
4. **Data Transfer**: Minimize data transfer costs by processing outputs on-instance
5. **AMI Preparation**: Create custom AMI with CUDA drivers pre-installed to save setup time

---

## Example Cost Scenarios

### Scenario 1: Small Simulation (0.5m spacing, ~12K receivers)
- **Instance**: g5.xlarge (Spot)
- **Duration**: 4 hours
- **Cost**: $0.15 × 4 = **$0.60**

### Scenario 2: Medium Simulation (0.1m spacing, ~600K receivers)
- **Instance**: g5.2xlarge (Spot)
- **Duration**: 12 hours
- **Cost**: $0.30 × 12 = **$3.60**

### Scenario 3: Large Simulation (0.05m spacing, ~4.7M receivers)
- **Instance**: g5.12xlarge (Spot)
- **Duration**: 24 hours
- **Cost**: $1.70 × 24 = **$40.80**

### Scenario 4: Same Large Simulation (On-Demand)
- **Instance**: g5.12xlarge (On-Demand)
- **Duration**: 24 hours
- **Cost**: $5.67 × 24 = **$136.08**

**Savings with Spot**: $95.28 (70% reduction)

---

*Note: Prices are approximate and vary by region and time. Check AWS Pricing Calculator for current rates.*
