?	?[;Q?O@?[;Q?O@!?[;Q?O@	ʖ?c????ʖ?c????!ʖ?c????"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:?[;Q?O@AG?Z?Q??A#???zO@Y?/fKVE??rEagerKernelExecute 0*	?Q??i@2U
Iterator::Model::ParallelMapV2V??L???!i?[MuD@)V??L???1i?[MuD@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?p??[u??!j(?P|A@)?fb????1&??7@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlices???M??![?-??%@)s???M??1[?-??%@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat|~!<??!?%???)@)?C????1.???j%@:Preprocessing2F
Iterator::Model1????4??!w?3d?H@)k{?????1:?$?? @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??(???!?8̛?iI@)????!9y?1Ն$2d?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor???U?q?!-r?<@? @)???U?q?1-r?<@? @:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?o??R???!??:??zA@)??H?}]?1???%???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9ʖ?c????I5/N3?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	AG?Z?Q??AG?Z?Q??!AG?Z?Q??      ??!       "      ??!       *      ??!       2	#???zO@#???zO@!#???zO@:      ??!       B      ??!       J	?/fKVE???/fKVE??!?/fKVE??R      ??!       Z	?/fKVE???/fKVE??!?/fKVE??b      ??!       JCPU_ONLYYʖ?c????b q5/N3?X@Y      Y@qDf??Q??"?
device?Your program is NOT input-bound because only 0.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQ2"CPU: B 