using UnityEngine;

public class CarMover : MonoBehaviour
{
    private TargetWaypoint _targetWaypoint;
    private Transform currentWaypoint;
    private float moveSpeed;
    private float distanceToWaypoint = 0.1f;
    public int waypointOffset = 0;

    private void Awake()
    {
        _targetWaypoint = FindObjectOfType<TargetWaypoint>();

        // 1) 첫 번째 웨이포인트 가져오기
        Transform wp = _targetWaypoint.GetNextWayPoint(null);

        // 2) waypointOffset만큼 건너뛰기
        for (int i = 0; i < waypointOffset; i++)
        {
            var next = _targetWaypoint.GetNextWayPoint(wp);
            if (next == null) break;
            wp = next;
        }

        // 3) 그 위치로 순간이동
        transform.position = wp.position;

        // 4) 다음 웨이포인트를 향해 바라보고 속도 설정
        currentWaypoint = _targetWaypoint.GetNextWayPoint(wp);
        if (currentWaypoint != null)
        {
            transform.LookAt(currentWaypoint);
            moveSpeed = _targetWaypoint.GetWaypointSpeed(currentWaypoint);
        }
    }

    private void FixedUpdate()
    {
        if (currentWaypoint == null) return;

        transform.position = Vector3.MoveTowards(
            transform.position,
            currentWaypoint.position,
            moveSpeed * Time.deltaTime
        );

        if (Vector3.Distance(transform.position, currentWaypoint.position) < distanceToWaypoint)
        {
            currentWaypoint = _targetWaypoint.GetNextWayPoint(currentWaypoint);
            if (currentWaypoint != null)
            {
                moveSpeed = _targetWaypoint.GetWaypointSpeed(currentWaypoint);
                transform.LookAt(currentWaypoint);
            }
        }
    }
}



// using System.Collections;
// using UnityEngine;

// public class CarMover : MonoBehaviour
// {
//     private TargetWaypoint _targetWaypoint;
//     private float distanceToWaypoint = 0.1f;

//     private Transform currentWaypoint;
//     public Vector3 CurrentPosition { get; private set; }

//     private float moveSpeed; 

//     private void Awake()
//     {
//         _targetWaypoint = FindObjectOfType<TargetWaypoint>();

//         currentWaypoint = _targetWaypoint.GetNextWayPoint(null);
//         transform.position = currentWaypoint.position;

//         currentWaypoint = _targetWaypoint.GetNextWayPoint(currentWaypoint);
//         transform.LookAt(currentWaypoint);

//         moveSpeed = _targetWaypoint.GetWaypointSpeed(currentWaypoint);
//     }

//     private void FixedUpdate()
//     {
//         if (currentWaypoint == null)
//         {
//             currentWaypoint = _targetWaypoint.GetNextWayPoint(null);
//             transform.LookAt(currentWaypoint);
//         }

//         transform.position = Vector3.MoveTowards(transform.position, currentWaypoint.position, moveSpeed * Time.deltaTime);

//         if (Vector3.Distance(transform.position, currentWaypoint.position) < distanceToWaypoint)
//         {
//             currentWaypoint = _targetWaypoint.GetNextWayPoint(currentWaypoint);

//             if (currentWaypoint != null)
//             {
//                 moveSpeed = _targetWaypoint.GetWaypointSpeed(currentWaypoint);

//                 transform.LookAt(currentWaypoint);
//             }
//         }
//     }
// }
