	?KU??b!@?KU??b!@!?KU??b!@	H?K\??H?K\??!H?K\??"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:?KU??b!@Hm??~??AV?a?
!@Y?v/??Q??rEagerKernelExecute 0*	??S㥷j@2U
Iterator::Model::ParallelMapV2?:]????!??a$B@)?:]????1??a$B@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate7p??G??!???>?<@)gҦ?٬?1?b*?r\:@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?٬?\m??!?զ???:@)?
??捫?193?n?-9@:Preprocessing2F
Iterator::Model?aK??z??!???a??D@)?9y?	???1q?? 0@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipt??gy??!yJ?uuM@)I?"i7?x?13]a??@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice|???ss?!q?c?@)|???ss?1q?c?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor????a?m?!?%j?b??)????a?m?1?%j?b??:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?tZ?A???!???N?,=@):?`???T?1?V?;????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 1.3% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9I?K\??I?;?ӎ?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	Hm??~??Hm??~??!Hm??~??      ??!       "      ??!       *      ??!       2	V?a?
!@V?a?
!@!V?a?
!@:      ??!       B      ??!       J	?v/??Q???v/??Q??!?v/??Q??R      ??!       Z	?v/??Q???v/??Q??!?v/??Q??b      ??!       JCPU_ONLYYI?K\??b q?;?ӎ?X@