	?"r?@?"r?@!?"r?@	vK?=?@vK?=?@!vK?=?@"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:?"r?@?e??@???A???խ@YHlw?}??rEagerKernelExecute 0*	?z?G}`@2U
Iterator::Model::ParallelMapV2E)!XU/??!Ŗ?7	G@)E)!XU/??1Ŗ?7	G@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?????=??!??]J?|<@)??מY??1As?$??7@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatd??????!8s????&@)?c?~???1RD?+*"@:Preprocessing2F
Iterator::Model???	???!?W??|?K@)?Ø??R??1?G?"@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice_??x?Zy?!?n\??@)_??x?Zy?1?n\??@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip.?l?IF??!?[F?iF@)?Ia??Ls?1?\?9s?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?-X?xi?!ф????@)?-X?xi?1ф????@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap\Y???"??!?
=?)?=@)??V???\?1?????3??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 2.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9vK?=?@Il??FpX@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?e??@????e??@???!?e??@???      ??!       "      ??!       *      ??!       2	???խ@???խ@!???խ@:      ??!       B      ??!       J	Hlw?}??Hlw?}??!Hlw?}??R      ??!       Z	Hlw?}??Hlw?}??!Hlw?}??b      ??!       JCPU_ONLYYvK?=?@b ql??FpX@