?	?Q?=?.@?Q?=?.@!?Q?=?.@	??h;?????h;???!??h;???"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:?Q?=?.@it?3???Ai:;E.@Y??pz??rEagerKernelExecute 0*	??????N@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?U?6????!?????>@)N?q?
??1?_?zB?8@:Preprocessing2U
Iterator::Model::ParallelMapV2??W9҉?!0?F/?4@)??W9҉?10?F/?4@:Preprocessing2F
Iterator::Modelq?::?F??!B?h??A@)GW#???1?(U???-@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate???-c??!G??0?e7@)?_??s??1?????1*@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicel?????y?!?^?	?$@)l?????y?1?^?	?$@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?r?m?B??!ߎK|?!P@)&7??5t?1?F??` @:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?A?L??j?!??䰝u@)?A?L??j?1??䰝u@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?????!????{f:@)U???)^?1Pz??,@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.3% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9??h;???Is?2A?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	it?3???it?3???!it?3???      ??!       "      ??!       *      ??!       2	i:;E.@i:;E.@!i:;E.@:      ??!       B      ??!       J	??pz????pz??!??pz??R      ??!       Z	??pz????pz??!??pz??b      ??!       JCPU_ONLYY??h;???b qs?2A?X@Y      Y@qS?zF%o??"?
device?Your program is NOT input-bound because only 0.3% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
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