	??k?V@??k?V@!??k?V@	?W?X???W?X??!?W?X??"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:??k?V@??x????A???%:?V@Y?[?J???rEagerKernelExecute 0*	??v???f@2U
Iterator::Model::ParallelMapV2 ?)U????!k??P??G@) ?)U????1k??P??G@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateJ???nI??!??9_U@@)??4?䚪?1F?ш"?<@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatF?W?????!???R?k$@)R???0???16c???	!@:Preprocessing2F
Iterator::Model??0Bx??!dBh??dJ@)lЗ??\??1ȟ????@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice=e5]Ot}?!?U??@)=e5]Ot}?1?U??@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip????!???!???qu?G@)S?1?#y?1HR$Z3@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor? ??zi?!?,??)? ??zi?1?,??:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?=?-??!z??g'?@@)5?\??u\?1?5;_???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?W?X??Is?p?S?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??x??????x????!??x????      ??!       "      ??!       *      ??!       2	???%:?V@???%:?V@!???%:?V@:      ??!       B      ??!       J	?[?J????[?J???!?[?J???R      ??!       Z	?[?J????[?J???!?[?J???b      ??!       JCPU_ONLYY?W?X??b qs?p?S?X@