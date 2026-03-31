# Model Contract

## Primary Output

The skill's main artifact is a native lifecycle model `json_ordered` payload.

## Required `json_ordered` Skeleton

Minimum sections that must exist before strict validation:

- `lifeCycleModelDataSet`
- `lifeCycleModelDataSet.@xmlns`
- `lifeCycleModelDataSet.@xmlns:common`
- `lifeCycleModelDataSet.@xmlns:xsi`
- `lifeCycleModelDataSet.@version`
- `lifeCycleModelDataSet.@xsi:schemaLocation`
- `lifeCycleModelDataSet.lifeCycleModelInformation.dataSetInformation.common:UUID`
- `lifeCycleModelDataSet.lifeCycleModelInformation.dataSetInformation.name`
- `lifeCycleModelDataSet.lifeCycleModelInformation.dataSetInformation.classificationInformation`
- `lifeCycleModelDataSet.lifeCycleModelInformation.dataSetInformation.referenceToResultingProcess`
- `lifeCycleModelDataSet.lifeCycleModelInformation.quantitativeReference.referenceToReferenceProcess`
- `lifeCycleModelDataSet.lifeCycleModelInformation.technology.processes.processInstance`

## Required Process Instance Fields

Each `processInstance` must carry:

- `@dataSetInternalID`
- `@multiplicationFactor`
- `referenceToProcess.@refObjectId`
- `referenceToProcess.@type`
- `referenceToProcess.@uri`
- `referenceToProcess.@version`
- `connections.outputExchange`

Each `outputExchange` should carry:

- `@flowUUID`
- `downstreamProcess[*].@id`

## Validation Gates

Block the model if any of these are true:

- no reference process node is marked
- no referenced process row can be resolved
- the process graph cannot produce defensible upstream or downstream connections
- strict `tidas-sdk` validation fails
- `tidas-tools` classification validation fails

## Downstream Ownership

This skill does not emit:

- `json_tg`
- `rule_verification`
- generated resulting-process artifacts

If the user later approves writes, the downstream publish layer owns any platform-specific derivation from the native `json_ordered` payload.
