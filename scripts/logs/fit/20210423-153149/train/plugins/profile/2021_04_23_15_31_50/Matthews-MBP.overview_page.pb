?	dyW=`#@dyW=`#@!dyW=`#@	z6h?????z6h?????!z6h?????"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:dyW=`#@?z?Fw??A?aN?&?"@Yg??F??rEagerKernelExecute 0*	??K7?Ik@2F
Iterator::Model????????!.?z???O@)_B?D??1v????!B@:Preprocessing2U
Iterator::Model::ParallelMapV2??0Ӯ?!q?1K?;@)??0Ӯ?1q?1K?;@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??????!?:k}?E7@)n???Wu??1܇]?4@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?i>"???!?S??y"@)??S????1X`?r?@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?3?l|?!x?E?m	@)?3?l|?1x?E?m	@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip^????4??!??r	B@)kH?c?Cw?1:I???@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?'??Ql?!Q?N?|V??)?'??Ql?1Q?N?|V??:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap???w?-??!!L?.5Q8@)4?ތ??b?1V1????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 2.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9z6h?????I&_?	?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?z?Fw???z?Fw??!?z?Fw??      ??!       "      ??!       *      ??!       2	?aN?&?"@?aN?&?"@!?aN?&?"@:      ??!       B      ??!       J	g??F??g??F??!g??F??R      ??!       Z	g??F??g??F??!g??F??b      ??!       JCPU_ONLYYz6h?????b q&_?	?X@Y      Y@qu??i1???"?
device?Your program is NOT input-bound because only 2.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
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