	j????D@j????D@!j????D@	?F@??j@?F@??j@!?F@??j@"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:j????D@+?WY???A??xy:w@Y?Wt?5=??rEagerKernelExecute 0*	?/?$Ng@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateuYLl>???!?s?O?F@)?J???J??1?Ȥ??AE@:Preprocessing2U
Iterator::Model::ParallelMapV2s?蜟???!??f?H?A@)s?蜟???1??f?H?A@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?Ƽ?8d??!?ӪZhP$@)7??͏?1>6??? @:Preprocessing2F
Iterator::ModelF?N????!BQ??k?D@)?.???ǅ?1 $?Y?@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice????y7v?!??*?F@)????y7v?1??*?F@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipڐf??!??]f?uM@)ADj??4s?1???@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorM?n?k?!C??#???)M?n?k?1C??#???:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??)????!????G@)?[X7?Y?1G!?x?O??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 2.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?F@??j@I??;?tX@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	+?WY???+?WY???!+?WY???      ??!       "      ??!       *      ??!       2	??xy:w@??xy:w@!??xy:w@:      ??!       B      ??!       J	?Wt?5=???Wt?5=??!?Wt?5=??R      ??!       Z	?Wt?5=???Wt?5=??!?Wt?5=??b      ??!       JCPU_ONLYY?F@??j@b q??;?tX@