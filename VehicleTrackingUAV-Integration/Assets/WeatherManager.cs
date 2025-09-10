using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class WeatherManager : MonoBehaviour {
    public enum Weather {SUNNY, SNOW };
    public Weather currentWeather;
    public ParticleSystem snow;
    public float weather_time = 10f; // 날씨 바뀌는 간격
    public int next_weather; //랜덤하게 다음 날씨 지정

    void Start()
    {
        currentWeather = Weather.SUNNY; //시작은 맑은 날씨
        next_weather = 1; // 다음 날씨는 무조건 눈
    }

    public void ChangeWeather(Weather weatherType)
    {
        if (weatherType != this.currentWeather) {
            switch (weatherType) {
                case Weather.SUNNY: 
                    currentWeather = Weather.SUNNY;
                    this.snow.Stop();
                    break;
                case Weather.SNOW:
                    currentWeather = Weather.SNOW;
                    this.snow.Play();
                    break;
            }
        }
        //매개변수로 받은 날씨가 현재 날씨와 같지 않다면 매개변수로 받은 날씨로 변경해준다. 
    }

    void Update()
    {
        this.weather_time -= Time.deltaTime; //10초동안은 그 날씨 유지
        if(next_weather == 1) //다음 날씨가 '눈'이고
        {
            if (this.weather_time <= 0) //현재 날씨의 제한시간이 끝나면
            {
                next_weather = Random.Range(0, 2); //다음 날씨 계산(0 - 맑음, 1 - 눈)
                ChangeWeather(Weather.SNOW); //눈으로 바꿔줌
                weather_time = 10f; 
            }
        }
        if (next_weather == 0) //다음 날씨가 '맑음'이고
        {
            if (this.weather_time <= 0) //현재 날씨의 제한시간이 끝나면
            {
                next_weather = Random.Range(0, 2); //다음 날씨 계산(0 - 맑음, 1 - 눈)
                ChangeWeather(Weather.SUNNY); //맑음으로 바꿔줌
                weather_time = 10f;
            }
        }
    }
}