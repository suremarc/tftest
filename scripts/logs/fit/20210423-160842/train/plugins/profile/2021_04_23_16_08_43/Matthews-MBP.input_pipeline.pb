	y7R?X:@y7R?X:@!y7R?X:@	/{?y???/{?y???!/{?y???"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:y7R?X:@,f??!??AgE?D?:@Y?E???Ը?rEagerKernelExecute 0*	     ?a@2U
Iterator::Model::ParallelMapV2???S㥫?!0K???B@)???S㥫?10K???B@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?&1???!??w??A@)?&1???1??w??A@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat;?O??n??!Ad?W?,)@);?O??n??1Ad?W?,)@:Preprocessing2F
Iterator::ModelD?l?????!?
fI9 H@)???Q???16?n??$@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice{?G?zt?!H?>???@){?G?zt?1H?>???@:Preprocessing2T
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.4% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9/{?y???I?$??o?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	,f??!??,f??!??!,f??!??      ??!       "      ??!       *      ??!       2	gE?D?:@gE?D?:@!gE?D?:@:      ??!       B      ??!       J	?E???Ը??E???Ը?!?E???Ը?R      ??!       Z	?E???Ը??E???Ը?!?E???Ը?b      ??!       JCPU_ONLYY/{?y???b q?$??o?X@