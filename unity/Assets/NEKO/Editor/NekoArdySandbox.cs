#if UNITY_EDITOR
using System;
using System.IO;
using System.Linq;
using Newtonsoft.Json.Linq;
using UnityEditor;
using UnityEngine;

// 只在隔离场景中回放已保存的生成结果，不启动模型服务或 VRChat。
public class NekoArdySandbox : EditorWindow
{
    private GameObject actor, floor;
    private NekoArdyRigLab rig;
    private NekoArdyPoseLab receiver;
    private JObject pose;
    private int frame;
    private double nextFrame;
    private bool finishing;
    private string status="请选择已保存的 ARDY pose JSON";
    private bool legacy;

    [MenuItem("NEKO/ARDY/动作回放沙盒")]
    public static void Open() { GetWindow<NekoArdySandbox>("ARDY 动作沙盒"); }

    private void OnGUI()
    {
        EditorGUILayout.HelpBox("仅限 ArdyLab 隔离场景。回放固定素材，字幕不参与动作计时。",MessageType.Info);
        legacy=EditorGUILayout.Toggle("重现旧的姿势叠加错误",legacy);
        if(GUILayout.Button("选择动作 JSON 并播放")) {
            string path=EditorUtility.OpenFilePanel("选择包含 pose 或 local_rot_mats 的 JSON","","json");
            if(!string.IsNullOrEmpty(path)) {
                try { Play(path,legacy); } catch(Exception e) { StopPlayback(); status=e.Message; }
            }
        }
        if(GUILayout.Button("停止并清理临时角色")) StopPlayback();
        EditorGUILayout.LabelField(status,EditorStyles.wordWrappedLabel);
    }

    public void Play(string path,bool reproduceLegacy=false)
    {
        StopPlayback();
        var scene=UnityEngine.SceneManagement.SceneManager.GetActiveScene();
        if(EditorApplication.isPlayingOrWillChangePlaymode || !scene.path.StartsWith("Assets/NEKO/ArdyLab/"))
            throw new InvalidOperationException("请在编辑模式下打开 ArdyLab 隔离场景");
        var parsed=JObject.Parse(File.ReadAllText(path));
        pose=parsed["pose"] as JObject ?? parsed;
        if(!(pose["local_rot_mats"] is JArray rotations) || rotations.Count<2 || rotations.Count>600
            || !(pose["root_positions"] is JArray roots) || roots.Count!=rotations.Count
            || !(pose["foot_contacts"] is JArray contacts) || contacts.Count!=rotations.Count)
            throw new InvalidOperationException("需要长度相同的旋转、根位移和接触数组，最多600帧");
        var source=UnityEngine.Object.FindObjectsOfType<NekoArdyRigLab>(true)
            .FirstOrDefault(r=>r.targetAnimator!=null && r.targetAnimator.gameObject==r.gameObject);
        if(source==null) throw new InvalidOperationException("隔离场景中缺少可复制的实验角色");
        try {
            actor=Instantiate(source.gameObject); actor.name="ARDY_Sandbox_Preview";
            actor.hideFlags=HideFlags.DontSave;
            actor.transform.position=new Vector3(20,0,0);
            rig=actor.GetComponent<NekoArdyRigLab>(); receiver=actor.GetComponent<NekoArdyPoseLab>();
            rig.motionRoot=actor.transform; rig.navigationWriter=null; rig.enabled=true; rig.labApproved=true;
            rig.rootEnabled=true; rig.feetEnabled=true; rig.environmentMask=1;
            receiver.enabled=true; receiver.labEnabled=true; receiver.requireAuthority=false;
            receiver.authorityRouter=null; receiver.worldBridge=null; receiver.telemetry=null;
            receiver.allowIsolatedArm=true;
            NekoArdySetup.CaptureBindPose(rig);
            var animator=rig.targetAnimator;
            var idle=animator.runtimeAnimatorController.animationClips.FirstOrDefault(c=>c.name=="NarrowIdle");
            if(idle!=null) idle.SampleAnimation(actor,.5f);
            if(reproduceLegacy) rig.bindRotations=rig.targetBones.Select(b=>Quaternion.Inverse(actor.transform.rotation)*b.rotation).ToArray();
            floor=GameObject.CreatePrimitive(PrimitiveType.Cube); floor.name="ARDY_Sandbox_Floor";
            floor.hideFlags=HideFlags.DontSave; floor.transform.position=new Vector3(20,-.1f,0);
            floor.transform.localScale=new Vector3(12,.2f,12);
            Physics.SyncTransforms();
            receiver.ArmLab(); rig.Calibrate();
            if(receiver.stopped) throw new InvalidOperationException("标定失败："+rig.stopReason);
            frame=0; finishing=false; nextFrame=EditorApplication.timeSinceStartup;
            EditorApplication.update+=Tick;
            Selection.activeGameObject=actor;
            if(SceneView.lastActiveSceneView!=null) SceneView.lastActiveSceneView.Frame(new Bounds(actor.transform.position+Vector3.up*.9f,Vector3.one*2.4f),false);
            status="正在播放；来源为保存的生成结果";
        } catch { StopPlayback(); throw; }
    }

    private void Tick()
    {
        try {
            if(actor==null || EditorApplication.isPlayingOrWillChangePlaymode) { StopPlayback(); return; }
            if(!finishing && EditorApplication.timeSinceStartup>=nextFrame) {
                if(frame>=((JArray)pose["local_rot_mats"]).Count) finishing=true;
                else {
                    var root=pose["root_positions"][frame];
                    receiver.decodedRoot=new Vector3((float)root[0],(float)root[1],(float)root[2]);
                    receiver.frameMilliseconds=frame*50; receiver.lastSequence=frame/2+1;
                    receiver.approvedSequence=receiver.lastSequence; receiver.contactMask=0;
                    for(int c=0;c<4;c++) if((float)pose["foot_contacts"][frame][c]>=.5f) receiver.contactMask|=1<<c;
                    for(int j=0;j<27;j++) {
                        var m=pose["local_rot_mats"][frame][j];
                        receiver.decodedRotations[j]=Quaternion.LookRotation(new Vector3((float)m[0][2],(float)m[1][2],(float)m[2][2]),new Vector3((float)m[0][1],(float)m[1][1],(float)m[2][1]));
                    }
                    receiver.lastAcceptedAt=Time.realtimeSinceStartup;
                    frame+=2; nextFrame=EditorApplication.timeSinceStartup+.1;
                }
            }
            Physics.SyncTransforms(); rig.ApplyLatest();
            if(receiver.stopped) { string reason=rig.stopReason; StopPlayback(); status="回放已停止："+reason; }
            else if(finishing) {
                float distance=Vector3.Distance(actor.transform.position,new Vector3(20,0,0));
                EditorApplication.update-=Tick;
                rig.RestoreAnimator(); receiver.EmergencyStop();
                status="回放完成，根位移 "+distance.ToString("F3")+" 米；脚踝 IK 误差不代表鞋底无滑动。";
            }
            SceneView.RepaintAll(); Repaint();
        } catch(Exception e) { StopPlayback(); status="回放错误："+e.Message; }
    }

    public void StopPlayback()
    {
        EditorApplication.update-=Tick;
        if(rig!=null) rig.RestoreAnimator();
        if(actor!=null) DestroyImmediate(actor);
        if(floor!=null) DestroyImmediate(floor);
        actor=null; floor=null; rig=null; receiver=null;
        status="已停止";
    }
    private void OnDisable() { StopPlayback(); }
}
#endif
