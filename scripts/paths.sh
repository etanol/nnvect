nn_inputs=../sample_nn_data
knn_inputs=$nn_inputs
svm_inputs=../sample_svm_data
lin_inputs=$nn_inputs

output_base=../outputs
nn_outputs=$output_base/nn
knn_outputs=$output_base/knn
svm_outputs=$output_base/svm
lin_outputs=$output_base/linear

plot_base=../plots
nn_plots=$plot_base/nn
knn_plots=$plot_base/knn
svm_plots=$plot_base/svm
lin_plots=$plot_base/linear


ensure_path()
{
    if [ $# -gt 0 ]
    then
        test -d $1 || mkdir -p $1
    fi
}

