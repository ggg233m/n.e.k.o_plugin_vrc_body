using UdonSharp;
using UnityEngine;

// 独立面部通道只写经过编辑器校验的BlendShape，不触碰任何身体Transform。
[UdonBehaviourSyncMode(BehaviourSyncMode.None)]
public class NekoArdyFace : UdonSharpBehaviour
{
    public NekoNpcLocomotion locomotion;
    public bool configured;
    public SkinnedMeshRenderer[] renderers;
    public int[] shapeIndices;
    public float[] expressionValues;
    public int expressionCount;
    public int samplesPerCurve;
    public float[] durations;
    public float fadeSeconds=.15f;
    private float[] _from;
    private Mesh[] _meshes;
    private bool _wasOwned;
    private int _expression=-1;
    private float _weight=-1f,_started;
    private float _phase;

    public bool Ready()
    {
        if(!configured || locomotion==null || renderers==null || shapeIndices==null || expressionValues==null
            || renderers.Length==0 || renderers.Length>256 || shapeIndices.Length!=renderers.Length
            || expressionCount<1 || expressionCount>32 || samplesPerCurve<2 || samplesPerCurve>1201
            || durations==null || durations.Length!=expressionCount
            || expressionValues.Length!=expressionCount*renderers.Length*samplesPerCurve)return false;
        for(int i=0;i<durations.Length;i++)if(durations[i]<=0f || durations[i]>10f)return false;
        for(int i=0;i<renderers.Length;i++)
            if(renderers[i]==null || renderers[i].sharedMesh==null || shapeIndices[i]<0 || shapeIndices[i]>=renderers[i].sharedMesh.blendShapeCount)return false;
        return true;
    }

    private void LateUpdate()
    {
        if(locomotion==null || (!locomotion.externalPoseActive && !locomotion.poseTransitionActive)) {_wasOwned=false;return;}
        if(!_wasOwned) {
            if(!Ready())return;
            _from=new float[renderers.Length];_meshes=new Mesh[renderers.Length];
            for(int i=0;i<renderers.Length;i++)_meshes[i]=renderers[i].sharedMesh;
            _expression=-1;_wasOwned=true;_phase=0f;
        }
        int expression=Mathf.Clamp(locomotion.GetExpressionId(),0,expressionCount-1);
        float weight=Mathf.Clamp01(locomotion.GetExpressionWeight());
        for(int i=0;i<renderers.Length;i++)
            if(renderers[i]==null || renderers[i].sharedMesh!=_meshes[i]) {configured=false;_wasOwned=false;return;}
        if(_expression!=expression || Mathf.Abs(_weight-weight)>.0001f) {
            if(_expression!=expression)_phase=0f;
            _expression=expression;_weight=weight;_started=Time.realtimeSinceStartup;
            for(int i=0;i<renderers.Length;i++)_from[i]=renderers[i].GetBlendShapeWeight(shapeIndices[i]);
        }
        // 共用归一化时间，与现有1D表情混合树的片段时长混合一致。
        _phase=Mathf.Repeat(_phase+Time.unscaledDeltaTime/Mathf.Lerp(durations[0],durations[expression],weight),1f);
        float fade=Mathf.Clamp01((Time.realtimeSinceStartup-_started)/Mathf.Max(.001f,fadeSeconds));
        for(int i=0;i<renderers.Length;i++) {
            if(renderers[i]==null || renderers[i].sharedMesh!=_meshes[i]) {configured=false;_wasOwned=false;return;}
            float target=Mathf.Lerp(Sample(0,i),Sample(expression,i),weight);
            renderers[i].SetBlendShapeWeight(shapeIndices[i],Mathf.Lerp(_from[i],target,fade));
        }
    }

    private float Sample(int expression,int channel)
    {
        float position=_phase*(samplesPerCurve-1);
        int index=Mathf.Min(samplesPerCurve-2,Mathf.FloorToInt(position));
        int offset=(expression*renderers.Length+channel)*samplesPerCurve+index;
        return Mathf.Lerp(expressionValues[offset],expressionValues[offset+1],position-index);
    }
}
