using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class TrafficIntersection : MonoBehaviour
{
    [Range(0f, 2f)]
    [SerializeField]private float waypointSize = 1f;
    public List<TrafficIntersection> connectedIntersections; // 인접 교차로 리스트
    public float rightOffset = 2.0f; // 오른쪽으로 치우치는 정도
    private float arrowHeadLength = 1.0f;
    private float arrowHeadAngle = 20.0f;
    private float arrowOffset = 1.0f;

    private void OnDrawGizmos()
    {
        Gizmos.color = Color.white;
        Gizmos.DrawWireSphere(transform.position, waypointSize);
        
        if (connectedIntersections != null)
        {
            foreach (TrafficIntersection neighbor in connectedIntersections)
            {
                if (neighbor != null)
                {
                    DrawArrow(transform.position, neighbor.transform.position);
                }
            }
        }
    }

    private void DrawArrow(Vector3 from, Vector3 to)
    {
        Gizmos.color = Color.white;
        Vector3 direction = (to - from).normalized;
        
        // 오른쪽으로 치우치게 하는 오프셋 계산
        Vector3 rightOffsetVector = Vector3.Cross(Vector3.up, direction) * rightOffset;
        Vector3 adjustedFrom = from + rightOffsetVector;
        Vector3 adjustedTo = to + rightOffsetVector;
        Gizmos.DrawLine(adjustedFrom, adjustedTo);
        Vector3 arrowPosition = from + direction * arrowOffset;
        
        if (Mathf.Abs(direction.x) > Mathf.Abs(direction.z))
        {
            if (direction.x > 0)
                Gizmos.color = Color.red; // 오른쪽
            else
                Gizmos.color = Color.green; // 왼쪽
        }
        else
        {
            if (direction.z > 0)
                Gizmos.color = Color.blue; // 위쪽
            else
                Gizmos.color = Color.yellow; // 아래쪽
        }

        Vector3 right = Quaternion.LookRotation(direction) * Quaternion.Euler(0, 180 + arrowHeadAngle, 0) * Vector3.forward;
        Vector3 left = Quaternion.LookRotation(direction) * Quaternion.Euler(0, 180 - arrowHeadAngle, 0) * Vector3.forward;
        
        Gizmos.DrawLine(arrowPosition, arrowPosition + right * arrowHeadLength);
        Gizmos.DrawLine(arrowPosition, arrowPosition + left * arrowHeadLength);
    }

    public Transform GetNextWayPoint(Transform currentWaypoint)
    {
        if (currentWaypoint == null)
        {
            return transform.GetChild(0);
        }

        if (currentWaypoint.GetSiblingIndex()<transform.childCount-1)
        {
            return transform.GetChild(currentWaypoint.GetSiblingIndex()+1);
        }
        else
        {
            return transform.GetChild(0);
        }
    }

}
