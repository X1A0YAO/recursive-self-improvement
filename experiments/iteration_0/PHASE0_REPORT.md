# Phase 0 瀹為獙璁板綍锛氬熀纭€璁炬柦涓庡熀绾垮缓绔?
- 鏃ユ湡锛?026-11-04
- 鎵ц鑰咃細Codex锛堝伐鍏蜂笌娴佺▼锛夛紱妯″瀷鑷韩浠呬綔涓鸿璇勬祴瀵硅薄
- 鐘舵€侊細璇勬祴杩涜涓紙鏈枃浠跺湪鍩虹嚎璇勬祴瀹屾垚鍚庡畾绋匡級

## 1. 鐜

| 椤?| 鍊?|
|---|---|
| OS | Windows 11 瀹跺涵鐗堜腑鏂囩増 (10.0.26200), AMD64 |
| CPU | Intel Core i5-14400F (10C/16T) |
| RAM | 31.7 GB |
| GPU | NVIDIA GeForce RTX 4060 Ti 8GB (sm_89), CUDA 12.6, 椹卞姩 610.60 |
| Python | 3.12.15 (uv venv, .venv/) |
| PyTorch | 2.14.1+cu126 |
| transformers | 5.22.1 |
| 纾佺洏 | D: 651.6 GB锛堥」鐩墍鍦ㄧ洏锛屽墿浣?~288 GB锛?|

鍔犺浇鏂瑰紡锛欱F16 + `device_map="auto"`锛圙PU 5.99GB + CPU offload锛夛紝
璐績瑙ｇ爜锛坉o_sample=False锛変繚璇佸彲澶嶇幇銆?
## 2. 妯″瀷

- 鐩爣妯″瀷锛歈wen/Qwen3-4B锛堝師濮?HF 鏉冮噸锛孊F16 safetensors锛屾棤閲忓寲锛?- 涓嬭浇鏍￠獙锛氫笁涓垎鐗?SHA256 涓?HF API 杩斿洖鍊奸€愪竴绋嬪簭鍖栨瘮瀵癸紝鍏ㄩ儴涓€鑷?  - model-00001-of-00003.safetensors  3,957,900,840 bytes
  - model-00002-of-00003.safetensors  3,987,450,520 bytes
  - model-00003-of-00003.safetensors    99,630,640 bytes
- V0 checkpoint锛歝heckpoints/V0/锛堝彧璇诲揩鐓э紝SHA256 鍚屾牱鏍￠獙閫氳繃锛?- 鍙傛暟绱㈠紩锛歳esults/param_index_V0.json
  - 398 寮犻噺锛?.02B 鍙傛暟
  - attention 943.73M / MLP 2.69B / norm+embed 389.14M

## 3. 璇勬祴闆嗭紙benchmarks/manifest.json锛?
涓夌粍涓ユ牸闅旂锛堟瀯寤鸿剼鏈柇瑷€ prompt 闆堕噸鍙狅紱seed=20261104 鍥哄畾鍙鐜帮級锛?
| split | n | math | instruction | logic | repetition | general | 鐢ㄩ€?|
|-------|---|------|-------------|-------|------------|---------|------|
| optimization | 63 | 25 | 10 | 8 | 8 | 12 | Agent 鍙锛岃嚜鎴戝垎鏋愮敤 |
| validation | 59 | 25 | 10 | 6 | 6 | 12 | 淇敼鎺ュ彈/鎷掔粷鍒ゅ畾 |
| test | 79 | 40 | 12 | 6 | 6 | 15 | 鏈€缁堟姤鍛婏紝Agent 姘镐笉鍙 |

鏁版嵁婧愶細
- math: openai/gsm8k (main/test)
- instruction: google/IFEval (train) + 瀹樻柟 instructions.py 妫€鏌ュ櫒锛堜弗鏍兼ā寮忥級
- logic: 鏈湴浜哄伐鏍￠獙棰樺簱 24 棰橈紙纭畾鎬у叧閿瘝绛旀锛?- repetition: 鏈湴妯℃澘锛堣姹傚啓 N 鍙ヤ笉閲嶅鍙ュ瓙锛涘彞瀛愰噸澶嶇巼 + char-8gram 寰幆鐜囷級
- general: truthfulqa/truthful_qa (mc1)

璇勫垎瑙勫垯锛?- 鎴柇鍝嶅簲锛堢敓鎴愬埌 max_new_tokens 鏈粨鏉燂級涓€寰?0 鍒嗭紙淇濆畧锛?- math: 鍝嶅簲鏈€鍚庢暟瀛?vs gold锛?### 鍚庣殑鏈€缁堢瓟妗堬級
- instruction: IFEval 瀹樻柟涓ユ牸妫€鏌ュ櫒鍏ㄨ繃涓?1
- logic: 鍏抽敭璇嶅尮閰嶏紙澶氬厓绱犻渶鍏ㄥ嚭鐜帮級
- repetition: 1 - 鍙ュ瓙閲嶅鐜囷紱char8gram 寰幆鐜?> 0.3 鐩存帴 0
- general: 澶氶€夊瓧姣嶅搴旈€夐」 == mc1 姝ｇ‘閫夐」

## 4. 鎺ㄧ悊閰嶇疆

- chat template锛堥潪鎬濊€冩ā寮忥紝enable_thinking=False锛?- max_new_tokens=768, batch_size=4, 宸?padding, 璐績瑙ｇ爜

## 5. 鍩虹嚎缁撴灉

锛堣瘎娴嬪畬鎴愬悗濉啓锛?
## 6. 閬囧埌鐨勯棶棰樹笌瑙ｅ喅

1. HF snapshot_download 鍦ㄦ湰鏈轰唬鐞嗙幆澧冧笅闀胯繛鎺ヨ閲嶇疆銆佸弽澶嶅崱姝?   鈫?鏀圭敤 curl 鍘熺敓鏂偣缁紶锛?C -锛? 鍒嗙墖 SHA256 鏍￠獙锛屾垚鍔?2. GSM8K gold 鎻愬彇 bug锛氬垵鐗堟鍒欐彁鍙栦簡 <<...>> 涓棿璁＄畻鍊艰€岄潪 #### 鍚庣殑鏈€缁堢瓟妗堬紝
   瀵艰嚧鍐掔儫娴嬭瘯 4/4 鍏?0 鍒嗭紙妯″瀷瀹為檯鍏ㄥ锛?   鈫?淇涓?`####\s*number` 妯″紡锛涗慨澶嶅悗鍐掔儫 4/4 鍏?1
3. 鍗曟潯涓茶鐢熸垚杩囨參锛圕PU offload 涓?50 鍒嗛挓鏈畬鎴?10 鏉★級
   鈫?鎵归噺鐢熸垚锛?鏉?鎵癸級+ 澧為噺钀界洏 + 鏂偣缁瘎锛涢€熷害鎻愬崌鏁板€?4. 璇勬祴涓彂鐜版ā鍨嬬埍鐢?Markdown/LaTeX 闀挎€濈淮閾撅紝256 tokens 浼氳鎴柇
   鈫?max_new_tokens 鎻愰珮鍒?768锛屾埅鏂 0 鍒嗗苟璁板綍

## 7. 缁撹涓庝笅涓€姝?
- Phase 0 鐩爣锛堢幆澧?妯″瀷/绱㈠紩/璇勬祴闆?鍩虹嚎/V0 蹇収锛夊叏閮ㄨ揪鎴?- 涓嬩竴姝ワ細绛夊緟鐢ㄦ埛纭鍚庤繘鍏?Phase 1锛堝彈鎺х己闄蜂笌鍙慨鏀规€ч獙璇侊級
- Phase 0 鏈熼棿鏈浠讳綍鏉冮噸鍋氫慨鏀癸紙read-only 浣跨敤妯″瀷锛?