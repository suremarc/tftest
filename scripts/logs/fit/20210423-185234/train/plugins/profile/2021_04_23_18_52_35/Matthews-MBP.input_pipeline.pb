	T???%@T???%@!T???%@	????e\??????e\??!????e\??"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:T???%@??^zo??A9?	?ʬ$@Yst??%??rEagerKernelExecute 0*	?A`??*f@2U
Iterator::Model::ParallelMapV2?-u?׃??!9?
?OJC@)?-u?׃??19?
?OJC@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??)?ޭ?!i6?r@@)$???~???1?C????=@:Preprocessing2F
Iterator::Model?p?{????!2?LMr"J@)?q??ۘ?1?	??`+@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?ᔹ?F??!??dU&@)'???????1?~\??h"@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice:vP??w?!??}?<p	@):vP??w?1??}?<p	@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipz?rK???!?????G@)?<+i?7t?1??|?_D@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?!??l?!?%?i~c??)?!??l?1?%?i~c??:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?Z?[!???!Q?  ?@@)??.??Y?1$j?CK??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 1.3% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9????e\??I t?h??X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??^zo????^zo??!??^zo??      ??!       "      ??!       *      ??!       2	9?	?ʬ$@9?	?ʬ$@!9?	?ʬ$@:      ??!       B      ??!       J	st??%??st??%??!st??%??R      ??!       Z	st??%??st??%??!st??%??b      ??!       JCPU_ONLYY????e\??b q t?h??X@