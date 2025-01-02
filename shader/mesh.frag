#version 330 core

in vec3 FragPos;
in vec3 Normal;
in float Height;

out vec4 FragColor;

uniform vec3 light_pos;
uniform mat3 I_light;
uniform mat3 K_materials;
uniform float shininess;
uniform int mode;

vec3 heatmap(float t) {
    vec3 color;

    if (t <= 0.0) {
        color = vec3(0.0, 0.0, 1.0); // Blue
    }
    else if (t <= 0.25) {
        color = mix(vec3(0.0, 0.0, 1.0), vec3(0.0, 1.0, 1.0), t * 4.0); // Blue to Cyan
    }
    else if (t <= 0.5) {
        color = mix(vec3(0.0, 1.0, 1.0), vec3(0.0, 1.0, 0.0), (t - 0.25) * 4.0); // Cyan to Green
    }
    else if (t <= 0.75) {
        color = mix(vec3(0.0, 1.0, 0.0), vec3(1.0, 1.0, 0.0), (t - 0.5) * 4.0); // Green to Yellow
    }
    else if (t <= 1.0) {
        color = mix(vec3(1.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0), (t - 0.75) * 4.0); // Yellow to Red
    }
    else {
        color = vec3(1.0, 0.0, 0.0); // Red
    }

    return color;
}


void main()
{
    // Normalize the height value based on the mesh's y-range
    // You might need to adjust these values based on your actual y-range
    float normalizedHeight = (Height - (-1.0)) / (2.0); // Assuming y range is [-1, 1]
    vec3 heatmapColor = heatmap(normalizedHeight);
    
    // Simple lighting
    vec3 lightDir = normalize(light_pos - FragPos);
    float diff = max(dot(normalize(Normal), lightDir), 0.0);
    vec3 diffuse = diff * vec3(0.7);
    
    vec3 result = heatmapColor * (diffuse + 0.3); // Adding some ambient
    FragColor = vec4(result, 1.0);
}