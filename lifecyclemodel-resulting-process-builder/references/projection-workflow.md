# Projection Workflow

## Objective

Turn one lifecycle model into one or more process datasets whose exchange values are calculated from the model graph.

## Stages

### 1. Intake
Normalize:
- lifecycle model id/version
- lifecycle model `json_ordered`
- optional `json_tg`
- projection mode (`primary-only` | `all-subproducts`)
- metadata override policy
- publish intent

### 2. Validation
Validate lifecycle model input before projection.

### 3. Process lookup
Resolve all `processInstance[*].referenceToProcess` rows required for projection calculations.

### 4. Graph construction
Construct graph edges from:
- `technology.processes.processInstance`
- `connections.outputExchange.downstreamProcess`

### 5. Scaling and dependence
Determine:
- reference process instance
- reference exchange direction
- dependence propagation
- node scaling factors

### 6. Allocation and final-product grouping
Use allocation logic to derive:
- balanced / unbalanced edges
- remaining exchanges
- primary resulting process
- secondary/subproduct resulting processes

### 7. Process packaging
For each projected resulting process:
- assemble process `json_ordered`
- stamp projected metadata
- retain relation to source model

### 8. Artifact preparation
Prepare:
- process payload bundle
- relation payloads
- projection report
- optional preview asset references

### 9. Publish handoff
Only when explicitly approved:
- write projected process rows
- write model/resulting-process relation metadata
- never delete existing rows
