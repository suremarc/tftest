	?q?煉'@?q?煉'@!?q?煉'@	?*?S'|???*?S'|??!?*?S'|??"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:?q?煉'@Q?[????Aǜg?K.'@Y%??R???rEagerKernelExecute 0*	|?5^?mf@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate`??"????!??n???D@)J&?v???1?ɝC@:Preprocessing2U
Iterator::Model::ParallelMapV2?N?o+??!I??X??B@)?N?o+??1I??X??B@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatW??mU??!0??_??#@)X???ލ?1??n?A @:Preprocessing2F
Iterator::ModelO??????!??|??E@)??n/i???1ò*%f?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?s????!^??>L@)*Ŏơ~w?1S???	@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice????G6w?!x??9ND	@)????G6w?1x??9ND	@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapy?t????!?`TL??E@)uv28J^m?1?Ƴ?????:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorZՒ?r0k?!??؂????)ZՒ?r0k?1??؂????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 1.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?*?S'|??IV??b?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	Q?[????Q?[????!Q?[????      ??!       "      ??!       *      ??!       2	ǜg?K.'@ǜg?K.'@!ǜg?K.'@:      ??!       B      ??!       J	%??R???%??R???!%??R???R      ??!       Z	%??R???%??R???!%??R???b      ??!       JCPU_ONLYY?*?S'|??b qV??b?X@