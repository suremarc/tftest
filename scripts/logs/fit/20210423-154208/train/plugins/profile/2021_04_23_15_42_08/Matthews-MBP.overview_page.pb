?	?s}@?s}@!?s}@	"Άn????"Άn????!"Άn????"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:?s}@߉Y/?r??A???߾N@Y?n??Ia??rEagerKernelExecute 0*	??x?&qe@2U
Iterator::Model::ParallelMapV2????9??!?\
4?C@)????9??1?\
4?C@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateWya?X??!
??4??@@)&R???0??1?_?SH?;@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMape?/?????!??g?<E@)?-t%Տ?1,<(?D"@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatZ?>?-W??!???4??!@)j?TQ?ʊ?1??k?@:Preprocessing2F
Iterator::Model+??-??!9;??F@)?zi? ???1뱅AH?@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceÜ?M???!???V?z@)Ü?M???1???V?z@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??׻??!????K@)?????r?1?TAP?4@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?K?K?1b?!@???b???)?K?K?1b?1@???b???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 1.4% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9"Άn????I??E?E?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	߉Y/?r??߉Y/?r??!߉Y/?r??      ??!       "      ??!       *      ??!       2	???߾N@???߾N@!???߾N@:      ??!       B      ??!       J	?n??Ia???n??Ia??!?n??Ia??R      ??!       Z	?n??Ia???n??Ia??!?n??Ia??b      ??!       JCPU_ONLYY"Άn????b q??E?E?X@Y      Y@qbf p?e??"?
device?Your program is NOT input-bound because only 1.4% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
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