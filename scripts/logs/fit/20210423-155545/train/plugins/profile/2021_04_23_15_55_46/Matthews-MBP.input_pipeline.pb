	?C?l??@?C?l??@!?C?l??@	o?/?N@o?/?N@!o?/?N@"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:?C?l??@?E???Ը?A?n??J@Yw??/???rEagerKernelExecute 0*	      x@2F
Iterator::Model???K7???!??????Q@)Zd;?O???1VUUUU?G@:Preprocessing2U
Iterator::Model::ParallelMapV2
ףp=
??!     p7@)
ףp=
??1     p7@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap
ףp=
??!     p7@)
ףp=
??1     p7@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???Q???!     @@)???Q???1     @@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice{?G?zt?!VUUUU???){?G?zt?1VUUUU???:Preprocessing2T
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 4.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9o?/?N@I????W@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?E???Ը??E???Ը?!?E???Ը?      ??!       "      ??!       *      ??!       2	?n??J@?n??J@!?n??J@:      ??!       B      ??!       J	w??/???w??/???!w??/???R      ??!       Z	w??/???w??/???!w??/???b      ??!       JCPU_ONLYYo?/?N@b q????W@