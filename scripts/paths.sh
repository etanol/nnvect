nn_inputs=../sample_nn_data
knn_inputs=$nn_inputs
cl_inputs=$nn_inputs
svm_inputs=../sample_svm_data
lin_inputs=$nn_inputs

output_base=../outputs
nn_outputs=$output_base/nn
knn_outputs=$output_base/knn
cl_outputs=$output_base/cl
svm_outputs=$output_base/svm
lin_outputs=$output_base/linear

plots=../plots


ensure_path()
{
    if [ $# -gt 0 ]
    then
        test -d $1 || mkdir -p $1
    fi
}

