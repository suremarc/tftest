	??yǄQ@??yǄQ@!??yǄQ@	?mb?8:???mb?8:??!?mb?8:??"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:??yǄQ@l??F????A?;??JuQ@YS??c${??rEagerKernelExecute 0*	?C?l??l@2U
Iterator::Model::ParallelMapV2???ދ/??!m??t??B@)???ދ/??1m??t??B@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?r?9>Z??!?5???;@)?B]©?1?GVݪ6@:Preprocessing2F
Iterator::Model????????!t\(W>?L@)??????1ؒ??3@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatG?ҿ$???!?q??%@)??n????1?.:U??!@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceF??}ȋ?!??2??@)F??}ȋ?1??2??@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip-z?mø?!??ר?*E@)?^??x?z?16RhW?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor:y?	?5r?!?4???!??):y?	?5r?1?4???!??:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?V??,???!??}??<@)???;V?1J??z??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?mb?8:??I?α?b?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	l??F????l??F????!l??F????      ??!       "      ??!       *      ??!       2	?;??JuQ@?;??JuQ@!?;??JuQ@:      ??!       B      ??!       J	S??c${??S??c${??!S??c${??R      ??!       Z	S??c${??S??c${??!S??c${??b      ??!       JCPU_ONLYY?mb?8:??b q?α?b?X@