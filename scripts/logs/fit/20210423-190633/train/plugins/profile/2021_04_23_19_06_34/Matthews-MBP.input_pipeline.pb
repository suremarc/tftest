	-??;??K@-??;??K@!-??;??K@	?4?+???4?+??!?4?+??"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:-??;??K@&4I,)w??Ai?7>?K@YJ)??????rEagerKernelExecute 0*	?G?z?t@2U
Iterator::Model::ParallelMapV2^f?(???!??g?Y?I@)^f?(???1??g?Y?I@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??I??о?!/g{j&B@)	R)v4??1fz0h?A@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat=?Е??!V??@)ҩ+??y??1?2?U<h@:Preprocessing2F
Iterator::Model^G??t??!??,??K@)??^?????1?ne?#?@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?g	2*|?!?̮$P? @)?g	2*|?1?̮$P? @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipf`X???!S?Q?S_F@)s+??X?z?1?Ł?????:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor5?\??ul?!?1?????)5?\??ul?1?1?????:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap???eN??!D+4pB@){g?UId_?1?k?|??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.4% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?4?+??I???K??X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	&4I,)w??&4I,)w??!&4I,)w??      ??!       "      ??!       *      ??!       2	i?7>?K@i?7>?K@!i?7>?K@:      ??!       B      ??!       J	J)??????J)??????!J)??????R      ??!       Z	J)??????J)??????!J)??????b      ??!       JCPU_ONLYY?4?+??b q???K??X@