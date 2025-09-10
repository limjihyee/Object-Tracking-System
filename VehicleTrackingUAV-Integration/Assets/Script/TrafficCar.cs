using System.Collections;
using UnityEngine;
using UnityEngine.Serialization;

public class TrafficCar : MonoBehaviour
{
    public float speed = 5.0f;
    public float rotationSpeed = 10.0f;
    [FormerlySerializedAs("TargetTrfficIntersection")] public TrafficIntersection targetTrafficIntersection;
    private TrafficIntersection _currentTrafficIntersection;
    private TrafficIntersection _previousTrafficIntersection;
    private bool isMoving = false;
    //todo ���ͼ��� rightOffset ����ϵ��� ����
    public float rightOffset = 2.0f; // ���������� ġ��ġ�� ����

    private void Start()
    {
        if (targetTrafficIntersection == null || targetTrafficIntersection.connectedIntersections.Count == 0)
        {
            Debug.LogError("���� ���� ������ ���� �ʿ�");
            enabled = false;
            return;
        }
        isMoving = true;
    }

    private void Update()
    {
        if (isMoving && targetTrafficIntersection != null)
        {
            Vector3 direction = (targetTrafficIntersection.transform.position - transform.position).normalized;
            Vector3 rightOffsetVector = Vector3.Cross(Vector3.up, direction) * rightOffset;
            Vector3 adjustedDirection = (targetTrafficIntersection.transform.position + rightOffsetVector - transform.position).normalized;
            
            Quaternion targetRotation = Quaternion.LookRotation(adjustedDirection);
            transform.rotation = Quaternion.Slerp(transform.rotation, targetRotation, rotationSpeed * Time.deltaTime);

            transform.position += adjustedDirection * speed * Time.deltaTime;

            if (Vector3.Distance(transform.position, targetTrafficIntersection.transform.position) < rightOffset)
            {
                isMoving = false;
                _previousTrafficIntersection = _currentTrafficIntersection;
                _currentTrafficIntersection = targetTrafficIntersection;
                StartCoroutine(MoveToNextIntersection());
            }
        }
    }

    private IEnumerator MoveToNextIntersection()
    {
        yield return new WaitForSeconds(Random.Range(1.0f, 3.0f)); // ���� ��� �ð�
        do
        {
            targetTrafficIntersection = _currentTrafficIntersection.connectedIntersections[Random.Range(0, _currentTrafficIntersection.connectedIntersections.Count)];
        } while (targetTrafficIntersection == _previousTrafficIntersection);
        isMoving = true;
    }
}