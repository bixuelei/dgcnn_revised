'''
not finish yet, the self designed cross attention structure
'''
class Cross_attention(nn.Module):
    ''' Multi-Head Attention module '''
    def __init__(self, args, demention_target, demention_source):
        super(Cross_attention,self).__init__()
        self.num_heads = args.num_heads

        d_k = int(args.self_encoder_latent_features/self.num_heads)
        d_v = int(args.self_encoder_latent_features/self.num_heads)
        
        self.w_qs = nn.Linear(demention_target, self.num_heads * d_k, bias=False)
        self.w_ks = nn.Linear(demention_source, self.num_heads * d_k, bias=False)
        self.w_vs = nn.Linear(demention_source, self.num_heads * d_v, bias=False)
        self.fc = nn.Linear(self.num_heads * d_v, demention_target, bias=False)

        self.attention = Attention()

        self.norm1 = nn.LayerNorm(self.num_heads * d_v, eps=1e-6)
        self.norm2 = nn.LayerNorm(demention_target, eps=1e-6)
        self.act = nn.ReLU()
        self.bn=nn.BatchNorm1d(64)


    def forward(self, q, k, v):     #[bs,features,target_points]

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head      #d_k dimention of every key     d_v: dimention of every value   
        b_size, n_q, n_k = q.size(0), q.size(1), k.size(1)      #n_q  target features dimention     n_k:source features dimention

        original = q

        q = self.w_qs(q).view(-1, n_q, n_head, d_k)     #[bs,4096,4,16]
        k = self.w_ks(k).view(-1, n_k, n_head, d_k)     #[bs,4096,4,16]
        v = self.w_vs(v).view(-1, n_k, n_head, d_v)     #[bs,4096,4,16]
        
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)       #[bs,4,4096,16] [bs,4,4096,16]  [bs,4,4096,16]

        # get b x n x k x dv
        q, _ = self.attention(q, k, v)   #[bs,4,4096,16]
        
        # b x k x ndv
        q = q.transpose(1, 2).contiguous().view(b_size, n_q, -1)                #[bs,4096,64]
        s = self.norm1(q)                                                 #[bs,4096,64]
        res = self.act(self.norm2( original- self.fc(s)))                           #[bs,4096,features]
        x=original+res
        return x



class Decoder_Layers(nn.Module):
    def __init__(self, n_layers,decoder_layer,eincoder):
        super(Decoder_Layers, self).__init__()
        self.num_layers=n_layers
        self.decoder_layer=decoder_layer
        self.self_attention=eincoder
        self.self_attention_layers = _get_clones(self.self_attention, self.num_layers)
        self.layer_decoders = _get_clones(self.decoder_layer, self.num_layers)
    def forward(self, target,source):
        for i in range(self.num_layers):
            source=self.self_attention_layers[i](source)
            target=self.layer_decoders[i](target,source)
        return target


class Transformer(nn.Module):
    def __init__(self,args,num_sequence_source,num_sequence_target):
        super(Transformer, self).__init__()
        self.num_layers=args.num_layers
        self.num_heads=args.num_heads
        self.encoder_layer=SA_Layer_Multi_Head(args,num_sequence_source)
        self.encoder_layers=SA_Layers(self.num_layers,self.encoder_layer)
        self.decoder=Cross_attention(args,num_sequence_target, num_sequence_source)
        self.decoder_layers=Decoder_Layers(self.num_layers,self.decoder,self.encoder_layer)
    def forward(self,source, target):
        source=self.encoder_layers(source)
        target=self.decoder_layers(target,source,source)
        return target    

