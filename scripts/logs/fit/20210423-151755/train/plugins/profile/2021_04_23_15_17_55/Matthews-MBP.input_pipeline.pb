	?A`??@?A`??@!?A`??@	?ne!o?@?ne!o?@!?ne!o?@"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:?A`??@6x_????AՓ?G??@Yf?%?????rEagerKernelExecute 0*	@5^?I?n@2U
Iterator::Model::ParallelMapV2ݵ?|г??!?|?.L@)ݵ?|г??1?|?.L@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate@i?QH2??!?M*=,w>@)??*???1??k?*<@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat????;??!??X?|?@)bg
?׈?1???|?@:Preprocessing2F
Iterator::Model?? ?X4??!?i?rzN@)c??	???1??S'@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice}?;l"3w?!?????h@)}?;l"3w?1?????h@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipU?-?????!?e?	??C@)?k?,	Ps?1?rq?d???:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorJΉ=??e?!:??J ??)JΉ=??e?1:??J ??:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?s'????!>wo?T!?@)????Z?1R*?E??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 2.4% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?ne!o?@I?????cX@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	6x_????6x_????!6x_????      ??!       "      ??!       *      ??!       2	Փ?G??@Փ?G??@!Փ?G??@:      ??!       B      ??!       J	f?%?????f?%?????!f?%?????R      ??!       Z	f?%?????f?%?????!f?%?????b      ??!       JCPU_ONLYY?ne!o?@b q?????cX@