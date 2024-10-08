## Reusing an already running cluster

### Overview

SCT allows running tests using a previously provisioned cluster. This helps in efficiently utilizing resources and reduces setup time for subsequent tests.

Reusing a cluster is primarily intended for local development and debugging. Scenarios that can benefit from it include:
- re-run a test without having to wait for a new cluster to be provisioned
- re-run a test with updated (or different) configuration
- update Scylla packages on an existing cluster and re-run the test

### Prerequisites

To provision a cluster for later reuse, the initial test run must be executed with the `post_behavior` configuration parameters set to `keep` value.

This can be done in the test configuration YAML file as:
```yaml
post_behavior_db_nodes: "keep"
post_behavior_loader_nodes: "keep"
post_behavior_monitor_nodes: "keep"
```
For K8s cluster the configuration parameter to use is:
```yaml
post_behavior_k8s_cluster: "keep"
```
Alternatively, `SCT_POST_BEHAVIOR` environment variables can be set before running the test, for each node type to be kept:
```bash
export SCT_POST_BEHAVIOR_DB_NODES=keep
export SCT_POST_BEHAVIOR_LOADER_NODES=keep
export SCT_POST_BEHAVIOR_MONITOR_NODES=keep
```
For K8S cluster the environment variable to use is:
```bash
export SCT_POST_BEHAVIOR_K8S_CLUSTER=keep
```

### Configuration

Specify a cluster for reuse by adding its identifier to the SCT test configuration.
The identifier can be found in a few places:
- Jenkins logs of a build that provisioned the cluster - look for `test_id` attribute
- test run details in Argus - look for `Id` attribute on `Details` tab of the run
- console of a backend (AWS/GCE/Azure) where the cluster is provisioned - look for `TestId` tag of a cluster instance
- if the initial test run was executed locally, look for `test_id` attribute in test results inventory, e.g. `cat ~/sct-results/latest/test_id`

To specify which cluster to reuse set `reuse_cluster` parameter to `test_id` value in test configuration YAML file:
```yaml
reuse_cluster: 7c86f6de-f87d-45a8-9e7f-61fe7b7dbe84
```
Alternatively, set `SCT_REUSE_CLUSTER` environment variable before running the test:
```bash
export SCT_REUSE_CLUSTER=7c86f6de-f87d-45a8-9e7f-61fe7b7dbe84
```

### Example scenario of reusing a cluster

Example demonstrates reusing a cluster for updating Scylla packages on DB nodes and then re-running the initial test:
```bash
# initial test run
export SCT_POST_BEHAVIOR_DB_NODES=keep
export SCT_POST_BEHAVIOR_LOADER_NODES=keep
export SCT_POST_BEHAVIOR_MONITOR_NODES=keep
hydra run-test longevity_test.LongevityTest.test_custom_time --backend aws --config test-cases/longevity/longevity-10gb-3h.yaml --config configurations/network_config/test_communication_public.yaml

# reuse the cluster and update scylla packages (the new packages are placed on a local machine)
export SCT_REUSE_CLUSTER=$(cat ~/sct-results/latest/test_id)
hydra update-scylla-packages --test-id $SCT_REUSE_CLUSTER -p ~/new_scylla_packages --backend aws

# reuse the cluster and re-run the initial test
hydra run-test longevity_test.LongevityTest.test_custom_time --backend aws --config test-cases/longevity/longevity-10gb-3h.yaml --config configurations/network_config/test_communication_public.yaml
```
