	9?]??&@9?]??&@!9?]??&@	?MsH?	???MsH?	??!?MsH?	??"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:9?]??&@Y???"??Ac???&?&@Y???5????rEagerKernelExecute 0*	33333?N@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?C?r?c??! Se?]?>@)|`?? ??1D-yQq9@:Preprocessing2U
Iterator::Model::ParallelMapV2:???????!?=71?8@):???????1?=71?8@:Preprocessing2F
Iterator::Model?\6:秘?!?d$??rC@)-?}́?1g?r?,@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??3?ތ??!X?i??4@)aTR'????1i??(*@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice[&??|t?!J'#?Ǘ@)[&??|t?1J'#?Ǘ@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?|#?g]??!???N@)g&?5?p?1???4?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor????j?!s???1?@)????j?1s???1?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapdX??G??!?(Z???7@)??????]?1yQqJ??@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.4% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?MsH?	??I?????X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	Y???"??Y???"??!Y???"??      ??!       "      ??!       *      ??!       2	c???&?&@c???&?&@!c???&?&@:      ??!       B      ??!       J	???5???????5????!???5????R      ??!       Z	???5???????5????!???5????b      ??!       JCPU_ONLYY?MsH?	??b q?????X@