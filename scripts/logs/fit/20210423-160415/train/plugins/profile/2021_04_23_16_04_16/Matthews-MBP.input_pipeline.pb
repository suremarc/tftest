	1?Zd/@1?Zd/@!1?Zd/@	?Y`P????Y`P???!?Y`P???"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:1?Zd/@sh??|???A?$??/@Y?A`??"??rEagerKernelExecute 0*	     @g@2U
Iterator::Model::ParallelMapV2D?l?????!?1?c?B@)D?l?????1?1?c?B@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate???x?&??!,??B@)X9??v???1??????@@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat;?O??n??![k???Z3@);?O??n??1[k???Z3@:Preprocessing2F
Iterator::Model#??~j???!?c?1?E@)?~j?t???1?9??s?@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice{?G?zt?!?X`?@){?G?zt?1?X`?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapD?l?????!?1?c?B@)????Mb`?15?DM4??:Preprocessing2T
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.7% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?Y`P???IL??_??X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	sh??|???sh??|???!sh??|???      ??!       "      ??!       *      ??!       2	?$??/@?$??/@!?$??/@:      ??!       B      ??!       J	?A`??"???A`??"??!?A`??"??R      ??!       Z	?A`??"???A`??"??!?A`??"??b      ??!       JCPU_ONLYY?Y`P???b qL??_??X@