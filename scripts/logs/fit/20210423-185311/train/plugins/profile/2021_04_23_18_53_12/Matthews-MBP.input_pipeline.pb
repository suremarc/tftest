	ܠ?[;?$@ܠ?[;?$@!ܠ?[;?$@	B???(@B???(@!B???(@"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:ܠ?[;?$@J'L5???A??߼8)$@YI??r?S??rEagerKernelExecute 0*	;?O???s@2U
Iterator::Model::ParallelMapV2jg??R??![T???4E@)jg??R??1[T???4E@:Preprocessing2F
Iterator::Model2 {?????!L?ª$P@)?q??Q???1z??P?6@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??-Y???!?ڃ)??7@)?t???l??1Zy?5@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??X???!?߲?$@)??X???1?߲?$@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat.???1???!q???=#@)???׋?1߾DcNV@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice=b??Bw?!6?? ???)=b??Bw?16?? ???:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipe9	?/???!h1z???A@)?!H?v?1??<?'??:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap=c_??`??!?ww5"8@)Z?rL?_?1??s~{???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 2.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9A???(@I?C?NwX@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	J'L5???J'L5???!J'L5???      ??!       "      ??!       *      ??!       2	??߼8)$@??߼8)$@!??߼8)$@:      ??!       B      ??!       J	I??r?S??I??r?S??!I??r?S??R      ??!       Z	I??r?S??I??r?S??!I??r?S??b      ??!       JCPU_ONLYYA???(@b q?C?NwX@