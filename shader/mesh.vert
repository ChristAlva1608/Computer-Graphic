#version 330 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;

out vec3 FragPos;
out vec3 Normal;
out float Height;

uniform mat4 modelview;
uniform mat4 projection;

void main()
{
    FragPos = vec3(modelview * vec4(aPos, 1.0));
    Normal = mat3(transpose(inverse(modelview))) * aNormal;
    
    // Use the y-coordinate for height
    Height = aPos.y;
    
    gl_Position = projection * vec4(FragPos, 1.0);
}