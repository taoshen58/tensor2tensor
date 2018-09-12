import tensorflow as tf


"""
  hparams.add_hparam("encoder_self_attention_type", "none")

              attention_type=hparams.self_attention_type if hparams.encoder_self_attention_type.lower()=='none' else hparams.encoder_self_attention_type,

    elif attention_type == "mtsa_test":
      x = dot_product_attention_global(q, k, v, bias, dropout_rate, image_shapes)
"""

def multi_head_dense_layer(
  input_tensor_trans, hn, bias, bias_start=0.0,
  scope=None, dup_num=1, merge_var=False
):  # [bs,hd,sl,dim]
  with tf.variable_scope(scope, default_name='multi_head_dense_layer'):
    input_tensor = tf.transpose(input_tensor_trans, [1, 0, 2, 3])  # [bs,hd,sl,dim]-> [hd,bs,sl,dim]
    hd_num = input_tensor.get_shape().as_list()[0]
    bs = tf.shape(input_tensor)[1]
    sl = tf.shape(input_tensor)[2]
    hd_dim = input_tensor.get_shape().as_list()[3]

    if merge_var:
      weight = tf.get_variable('W', shape=[hd_num, hd_dim, hn * dup_num])
    else:
      weight_list = []
      for i in range(hd_num):
        sub_weight_list = []
        for j in range(dup_num):
          sub_weight_list.append(tf.get_variable('W_%d_%d' % (i, j), shape=[hd_dim, hn]))
        weight_list.append(tf.concat(sub_weight_list, -1) if dup_num > 1 else sub_weight_list[0])
      weight = tf.stack(weight_list, 0)
    input_tensor_rsp = tf.reshape(input_tensor, [hd_num, bs * sl, hd_dim])  # hd_num, bs*sl, hd_dim
    out_rsp = tf.matmul(input_tensor_rsp, weight)  # hd_num, bs*sl, hn
    if bias:
      if merge_var:
        bias_val = tf.get_variable('bias', shape=[hd_num, 1, hn], dtype=tf.float32,
                                   initializer=tf.constant_initializer(bias_start))
      else:
        bias_list = []
        for i in range(hd_num):
          sub_bias_list = []
          for j in range(dup_num):
            sub_bias_list.append(
                            tf.get_variable(
                                'bias_%d_%d' % (i, j), shape=[1, hn], dtype=tf.float32,
                                initializer=tf.constant_initializer(bias_start)))
          bias_list.append(tf.concat(sub_bias_list, -1) if dup_num > 1 else sub_bias_list[0])
        bias_val = tf.stack(bias_list, 0)
      out_rsp = out_rsp + bias_val
    out = tf.reshape(out_rsp, [hd_num, bs, sl, hn*dup_num])  # [hd,bs,sl,dim]
    return tf.transpose(out, [1, 0, 2, 3])  # [bs,hd,sl,dim]


def dot_product_attention_global(q,
                          k,
                          v,
                          bias,
                          dropout_rate=0.0,
                          image_shapes=None,
                          name=None,
                          make_image_summary=True):
  """dot-product attention.

  Args:
    q: a Tensor with shape [batch, heads, length_q, depth_k]
    k: a Tensor with shape [batch, heads, length_kv, depth_k]
    v: a Tensor with shape [batch, heads, length_kv, depth_v]
    bias: bias Tensor (see attention_bias())
    dropout_rate: a floating point number
    image_shapes: optional tuple of integer scalars.
      see comments for attention_image_summary()
    name: an optional string
    make_image_summary: True if you want an image summary.

  Returns:
    A Tensor.
  """
  with tf.variable_scope(
      name, default_name="dot_product_attention", values=[q, k, v]):
    # [batch, num_heads, query_length, memory_length]
    dim_q = q.get_shape().as_list()[-1]
    dim_k = k.get_shape().as_list()[-1]
    dim_v = v.get_shape().as_list()[-1]
    
    # token2token self attention
    dot_logits = tf.matmul(q, k, transpose_b=True)
    if bias is not None:
      dot_logits += bias
    e_dot_logits = tf.exp(dot_logits)  # bs,hd,ql,vl

    # source2token self-attention
    multi_logits = multi_head_dense_layer(k, dim_v, True, 0., 'multi_logits1')
    e_multi_logits = tf.exp(multi_logits)  # tf.nn.sigmoid(multi_logits)    # bs,hd,vl,vd

    # mtsa
    accum_z_deno = tf.matmul(e_dot_logits, e_multi_logits)  # bs,hd,ql,vd
    accum_z_deno = tf.where(  # in case of NaN and Inf
       tf.greater(accum_z_deno, tf.zeros_like(accum_z_deno)),
       accum_z_deno,
      tf.ones_like(accum_z_deno)
    )
    # attention dropout
    e_dot_logits = tf.nn.dropout(e_dot_logits, math.sqrt(1.-dropout_rate))
    e_multi_logits = tf.nn.dropout(e_multi_logits, math.sqrt(1-dropout_rate))
    rep_mul_score = v * e_multi_logits  # bs,hd,vl,vd
    accum_rep_mul_score = tf.matmul(e_dot_logits, rep_mul_score)  # bs,hd,ql,vd
    # calculate the final attention results
    attn_res = accum_rep_mul_score / accum_z_deno
    
    # for visualization
    weights = e_multi_logits / tf.reduce_sum(e_multi_logits, axis=-1, keepdims=True, name="attention_weights")
    if (not tf.get_variable_scope().reuse and
        # Summaries don't work well within tf.while_loop()
        "/while/" not in tf.contrib.framework.get_name_scope() and
        make_image_summary):
      attention_image_summary(weights, image_shapes)
    return attn_res
