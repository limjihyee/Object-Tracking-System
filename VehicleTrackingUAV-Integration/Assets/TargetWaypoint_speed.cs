using UnityEngine;

public class Waypoint : MonoBehaviour
{
    [Range(2.5f, 4.5f)]
    public float minSpeed = 2.5f; // �ӵ��� �ּҰ�

    [Range(2.5f, 4.5f)]
    public float maxSpeed = 3.5f; // �ӵ��� �ִ밪

    public float speed; // ���� ��������Ʈ�� �ӵ�

    private void Awake()
    {
        SetRandomSpeed(); // �ӵ��� �ʱ�ȭ
    }

    // �ӵ��� ���� ������ �����ϰ� ����
    public void SetRandomSpeed()
    {
        speed = Random.Range(minSpeed, maxSpeed);
    }

    // ���� ��������Ʈ�� �ӵ��� ��ȯ
    public float GetSpeed()
    {
        return speed;
    }
}
