?	/5B?S?%@/5B?S?%@!/5B?S?%@	|k?r9???|k?r9???!|k?r9???"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:/5B?S?%@?PMI????A??M?%@YQ?i>"??rEagerKernelExecute 0*	h??|?]g@2U
Iterator::Model::ParallelMapV2\??AA)??!{?@(?B@)\??AA)??1{?@(?B@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?[w?T???!^?Gwws5@)???v?Ӣ?1????3@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate1E?4~???!??d?.?6@)%???????1????8?2@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip::ZՒ??!?T?P??K@)??"?????1B???}$@:Preprocessing2F
Iterator::ModelU?2?F??!??;F@)??3????1???w?
@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice`x%?s}?!?6"?s@)`x%?s}?1?6"?s@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?]K?=k?!^&?xXv??)?]K?=k?1^&?xXv??:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?ek}?Ц?!d86???7@)$Di?]?1??5z):??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 1.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9}k?r9???IR4?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?PMI?????PMI????!?PMI????      ??!       "      ??!       *      ??!       2	??M?%@??M?%@!??M?%@:      ??!       B      ??!       J	Q?i>"??Q?i>"??!Q?i>"??R      ??!       Z	Q?i>"??Q?i>"??!Q?i>"??b      ??!       JCPU_ONLYY}k?r9???b qR4?X@Y      Y@qpK.?e??"?
device?Your program is NOT input-bound because only 1.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
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