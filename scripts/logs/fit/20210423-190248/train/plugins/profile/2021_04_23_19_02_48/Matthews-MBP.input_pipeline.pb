	9Dܜ?#@9Dܜ?#@!9Dܜ?#@	|)lJ}??|)lJ}??!|)lJ}??"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:9Dܜ?#@#??u???A??7?y#@Y-????Ʒ?rEagerKernelExecute 0*	??C?Pg@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate????S???!??٣??B@)?]?????1!??	?@@:Preprocessing2U
Iterator::Model::ParallelMapV2 ?H? ??!??[/?;@@) ?H? ??1??[/?;@@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?????!??Hhd	5@)?dT8??1u?(H3@:Preprocessing2F
Iterator::Model?-?l?I??!*?l?T&C@)/?.?H??1?y???U@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?b.?z?!XԂ?@)?b.?z?1XԂ?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??[?v??!?a?\??N@)??b??u?1?,???@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor]S ???m?!??"?Q??)]S ???m?1??"?Q??:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??g\8??!%PH?B@)P?i4?X?1k??????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.9% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9{)lJ}??I?'k??X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	#??u???#??u???!#??u???      ??!       "      ??!       *      ??!       2	??7?y#@??7?y#@!??7?y#@:      ??!       B      ??!       J	-????Ʒ?-????Ʒ?!-????Ʒ?R      ??!       Z	-????Ʒ?-????Ʒ?!-????Ʒ?b      ??!       JCPU_ONLYY{)lJ}??b q?'k??X@