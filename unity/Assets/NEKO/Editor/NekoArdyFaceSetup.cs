#if UNITY_EDITOR
using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEditor;
using UnityEditor.Animations;
using UdonSharpEditor;

// 只提取现有面部曲线，并验证烘焙误差；含身体曲线的片段拒绝安装。
public static class NekoArdyFaceSetup
{
    public static string Configure(NekoArdyRigLab rig)
    {
        if(EditorApplication.isPlayingOrWillChangePlaymode || rig==null || rig.targetAnimator==null)
            throw new InvalidOperationException("需要编辑模式下已绑定的ARDY角色");
        var controller=rig.targetAnimator.runtimeAnimatorController as AnimatorController;
        if(controller==null)throw new InvalidOperationException("需要可检查的AnimatorController");
        var router=rig.receiver.authorityRouter;
        var states=controller.layers.SelectMany(l=>l.stateMachine.states).Select(s=>s.state).ToArray();
        var clips=new AnimationClip[router.expressionNames.Length];
        var neutralClips=new AnimationClip[clips.Length];
        for(int i=0;i<clips.Length;i++) {
            var state=states.Single(s=>s.name=="Expression_"+i);
            if(state.speed!=1f || state.speedParameterActive || state.timeParameterActive || state.cycleOffset!=0f || state.cycleOffsetParameterActive)
                throw new InvalidOperationException("表情状态有未支持的时序控制");
            if(state.motion is AnimationClip)clips[i]=(AnimationClip)state.motion;
            else {
                var tree=state.motion as BlendTree;
                if(tree==null || tree.blendType!=BlendTreeType.Simple1D || tree.blendParameter!="ExpressionWeight"
                    || tree.children.Length!=2 || tree.children[0].threshold!=0 || tree.children[1].threshold!=1
                    || tree.children.Any(c=>c.timeScale!=1f || c.cycleOffset!=0f || c.mirror))
                    throw new InvalidOperationException("不支持的表情混合树："+state.name);
                clips[i]=tree.children[1].motion as AnimationClip;
                neutralClips[i]=tree.children[0].motion as AnimationClip;
                if(neutralClips[i]==null)throw new InvalidOperationException("中性表情必须是可验证片段");
            }
            if(clips[i]==null || !clips[i].isLooping || clips[i].length<=0 || clips[i].length>10)
                throw new InvalidOperationException("仅支持有效的循环面部片段");
        }
        for(int i=0;i<clips.Length;i++)
            if(neutralClips[i]!=null && neutralClips[i]!=clips[0])
                throw new InvalidOperationException("表情混合树中性基准不一致");
        var values=new Dictionary<string,AnimationCurve>[clips.Length];
        var bindings=new Dictionary<string,EditorCurveBinding>();
        for(int i=0;i<clips.Length;i++) {
            values[i]=new Dictionary<string,AnimationCurve>();
            foreach(var binding in AnimationUtility.GetCurveBindings(clips[i])) {
                if(binding.type!=typeof(SkinnedMeshRenderer) || !binding.propertyName.StartsWith("blendShape."))
                    throw new InvalidOperationException("表情片段含非BlendShape曲线："+clips[i].name);
                var curve=AnimationUtility.GetEditorCurve(clips[i],binding);
                string key=binding.path+"/"+binding.propertyName;
                bindings[key]=binding;values[i][key]=curve;
            }
        }
        var keys=bindings.Keys.OrderBy(k=>k,StringComparer.Ordinal).ToArray();
        if(keys.Length==0 || keys.Length>256)throw new InvalidOperationException("面部曲线数量无效");
        int samples=1+Mathf.CeilToInt(clips.Max(c=>c.length)*120);
        var renderers=new SkinnedMeshRenderer[keys.Length];var indices=new int[keys.Length];var bank=new float[keys.Length*clips.Length*samples];
        float maxError=0f;
        for(int j=0;j<keys.Length;j++) {
            var binding=bindings[keys[j]];var target=rig.targetAnimator.transform.Find(binding.path);
            renderers[j]=target==null?null:target.GetComponent<SkinnedMeshRenderer>();
            if(renderers[j]==null || renderers[j].sharedMesh==null)throw new InvalidOperationException("面部Renderer不存在："+binding.path);
            indices[j]=renderers[j].sharedMesh.GetBlendShapeIndex(binding.propertyName.Substring(11));
            if(indices[j]<0)throw new InvalidOperationException("面部BlendShape不存在："+binding.propertyName);
            if(!values[0].ContainsKey(keys[j]))throw new InvalidOperationException("中性片段未定义该通道："+keys[j]);
            for(int i=0;i<clips.Length;i++) {
                var curve=values[i].ContainsKey(keys[j])?values[i][keys[j]]:values[0][keys[j]];
                float duration=values[i].ContainsKey(keys[j])?clips[i].length:clips[0].length;
                int offset=(i*keys.Length+j)*samples;
                for(int k=0;k<samples;k++) {
                    float value=curve.Evaluate(duration*k/(samples-1));
                    if(float.IsNaN(value)||float.IsInfinity(value))throw new InvalidOperationException("面部曲线含无效数值");
                    bank[offset+k]=value;
                }
                for(int k=0;k<samples-1;k++)for(int sub=1;sub<4;sub++) {
                    float f=sub*.25f;
                    float error=Mathf.Abs(curve.Evaluate(duration*(k+f)/(samples-1))-Mathf.Lerp(bank[offset+k],bank[offset+k+1],f));
                    maxError=Mathf.Max(maxError,error);
                    if(error>.25f)throw new InvalidOperationException("面部曲线烘焙误差超过0.25："+keys[j]);
                }
            }
        }
        // 上面全部验证成功后才修改场景，调用方负责在安装前备份并在安装后保存。
        var face=rig.GetComponent<NekoArdyFace>();
        if(face==null)face=UdonSharpComponentExtensions.AddUdonSharpComponent<NekoArdyFace>(rig.gameObject);
        Undo.RecordObject(face,"配置ARDY独立面部通道");
        face.locomotion=router.locomotion;face.renderers=renderers;face.shapeIndices=indices;
        face.expressionValues=bank;face.expressionCount=clips.Length;face.configured=true;
        face.samplesPerCurve=samples;face.durations=clips.Select(c=>c.length).ToArray();
        EditorUtility.SetDirty(face);UdonSharpEditorUtility.CopyProxyToUdon(face);
        return "表情="+clips.Length+"，BlendShape通道="+keys.Length+"，最大采样误差="+maxError;
    }
}
#endif
