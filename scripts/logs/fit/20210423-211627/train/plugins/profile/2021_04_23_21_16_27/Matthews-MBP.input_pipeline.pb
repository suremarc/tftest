	?? !ʇ!@?? !ʇ!@!?? !ʇ!@	A\??a???A\??a???!A\??a???"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:?? !ʇ!@y??M????A#??u/!@Y?T2 Tq??rEagerKernelExecute 0*	?? ?r<k@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?F?һ?!l??H@)6?e?s~??12q????G@:Preprocessing2U
Iterator::Model::ParallelMapV2]????۱?!Zw.@@)]????۱?1Zw.@@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatP?R)v??!ݙ5iW"@)M?Nϻ???1˭????@:Preprocessing2F
Iterator::Model??Wt?5??!????LC@)??0?ъ?1????	@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice???1ZGu?!?cz7?@)???1ZGu?1?cz7?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?]h??H??!.-??N@)R??m?t?1??????@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??l#n?!?N????)??l#n?1?N????:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapKu/3??!???+GI@)	?L?nX?1??????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 1.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9A\??a???I??Yy??X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	y??M????y??M????!y??M????      ??!       "      ??!       *      ??!       2	#??u/!@#??u/!@!#??u/!@:      ??!       B      ??!       J	?T2 Tq???T2 Tq??!?T2 Tq??R      ??!       Z	?T2 Tq???T2 Tq??!?T2 Tq??b      ??!       JCPU_ONLYYA\??a???b q??Yy??X@