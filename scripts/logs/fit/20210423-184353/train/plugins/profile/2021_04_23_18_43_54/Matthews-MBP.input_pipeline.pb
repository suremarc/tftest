	?&3?V4I@?&3?V4I@!?&3?V4I@	?1Q?????1Q????!?1Q????"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:?&3?V4I@?CP5z5??AK??I@YI?f??6??rEagerKernelExecute 0*	?x?&1?e@2U
Iterator::Model::ParallelMapV2??^EF??!Ư?;F@)??^EF??1Ư?;F@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateL?u?~??!M'?"d?>@)g??j+???1m?"?:?:@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?y8????!?aBX}'@)???q?ő?1?H?J??#@:Preprocessing2F
Iterator::ModelL8????!R?#*J@)??ԕ????1/??aB @:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice*?n?EE|?! g?K?@)*?n?EE|?1 g?K?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?L??ݴ?!?S??=pG@)????m3u?1???a?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?[X7?i?!??`?_6??)?[X7?i?1??`?_6??:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapfl?f???!?}V??@@)?Q,??b?1?E???U??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?1Q????Ig~???X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?CP5z5???CP5z5??!?CP5z5??      ??!       "      ??!       *      ??!       2	K??I@K??I@!K??I@:      ??!       B      ??!       J	I?f??6??I?f??6??!I?f??6??R      ??!       Z	I?f??6??I?f??6??!I?f??6??b      ??!       JCPU_ONLYY?1Q????b qg~???X@