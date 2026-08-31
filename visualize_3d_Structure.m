%% Visualize localized CMEP atomic centers from CSV
% Edit only the configuration section when using another saved structure.
% This script never modifies the CSV or its companion metadata file.

script_path = mfilename('fullpath');
if isempty(script_path)
    project_dir = pwd;
else
    project_dir = fileparts(script_path);
end

%% Configuration
csv_path = fullfile(project_dir, 'localized_atoms', ...
    'cmep_localized_atoms.csv');

render_mode = "markers";       % "spheres" or "markers"
color_column = "likelihood_score";
minimum_likelihood = -Inf;      % -Inf displays every saved center
colormap_name = "magma";       % "magma" or any MATLAB colormap function
fixed_color = [1.00, 0.35, 0.12];
background_color = [0, 0, 0];
minimum_alpha = 0.30;           % Alpha at the lowest displayed likelihood

sphere_radius = [];             % Empty: 20% of saved minimum separation
sphere_resolution = 6;          % Higher is rounder but uses more GPU memory
marker_size = 18;               % Used only when render_mode = "markers"
length_unit = "nm";             % Overridden by companion JSON when available

camera_azimuth = -38;
camera_elevation = 35;
initial_view_padding = 0.08;    % Fractional physical margin around the data
initial_zoom_factor = 0.78;     % Less than 1 zooms out after fitting the view
figure_position = [80, 80, 1100, 900];
show_colorbar = true;

%% Load and validate the localized-center table
if ~isfile(csv_path)
    error('CMEP:MissingCSV', 'CSV file not found:\n%s', csv_path);
end

atom_table = readtable(csv_path, 'VariableNamingRule', 'preserve');
variable_names = string(atom_table.Properties.VariableNames);
required_columns = ["x", "y", "z"];
missing_columns = required_columns(~ismember(lower(required_columns), ...
    lower(variable_names)));
if ~isempty(missing_columns)
    error('CMEP:MissingColumns', ...
        'CSV is missing required column(s): %s', ...
        strjoin(missing_columns, ', '));
end

atom_positions_xyz = [ ...
    getNumericColumn(atom_table, variable_names, "x"), ...
    getNumericColumn(atom_table, variable_names, "y"), ...
    getNumericColumn(atom_table, variable_names, "z")];

has_color_column = strlength(color_column) > 0 && ...
    any(strcmpi(variable_names, color_column));
if has_color_column
    atom_color_values = getNumericColumn(atom_table, variable_names, color_column);
elseif isfinite(minimum_likelihood)
    error('CMEP:MissingColorColumn', ...
        'minimum_likelihood requires the CSV column "%s".', color_column);
else
    atom_color_values = ones(height(atom_table), 1);
end

valid_rows = all(isfinite(atom_positions_xyz), 2) & isfinite(atom_color_values);
selected_rows = valid_rows & atom_color_values >= minimum_likelihood;
if ~any(selected_rows)
    error('CMEP:NoAtoms', ...
        'No finite atomic centers satisfy minimum_likelihood = %.6g.', ...
        minimum_likelihood);
end

discarded_count = nnz(~selected_rows);
atom_positions_xyz = atom_positions_xyz(selected_rows, :);
atom_color_values = atom_color_values(selected_rows);
displayed_atom_table = atom_table(selected_rows, :);

%% Read optional physical metadata
[csv_folder, csv_name] = fileparts(csv_path);
metadata_path = fullfile(csv_folder, [csv_name, '.json']);
metadata = struct();
if isfile(metadata_path)
    try
        metadata = jsondecode(fileread(metadata_path));
    catch metadata_error
        warning('CMEP:MetadataReadFailed', ...
            'Could not read companion metadata: %s', metadata_error.message);
    end
end

if isfield(metadata, 'length_unit') && strlength(string(metadata.length_unit)) > 0
    length_unit = string(metadata.length_unit);
end
if isempty(sphere_radius)
    if isfield(metadata, 'minimum_atom_separation') && ...
            isfinite(metadata.minimum_atom_separation) && ...
            metadata.minimum_atom_separation > 0
        sphere_radius = 0.20 * double(metadata.minimum_atom_separation);
    else
        coordinate_span = max(atom_positions_xyz, [], 1) - ...
            min(atom_positions_xyz, [], 1);
        positive_span = coordinate_span(coordinate_span > 0);
        if isempty(positive_span)
            sphere_radius = 1;
        else
            sphere_radius = 0.005 * max(positive_span);
        end
        warning('CMEP:EstimatedRadius', ...
            ['No minimum separation was found in companion metadata. ', ...
             'Using a display-only radius of %.6g %s.'], ...
            sphere_radius, length_unit);
    end
end

validateattributes(sphere_radius, {'numeric'}, ...
    {'scalar', 'real', 'finite', 'positive'}, mfilename, 'sphere_radius');
validateattributes(sphere_resolution, {'numeric'}, ...
    {'scalar', 'integer', '>=', 3}, mfilename, 'sphere_resolution');
validateattributes(marker_size, {'numeric'}, ...
    {'scalar', 'real', 'finite', 'positive'}, mfilename, 'marker_size');
validateattributes(minimum_alpha, {'numeric'}, ...
    {'scalar', 'real', 'finite', '>=', 0, '<=', 1}, ...
    mfilename, 'minimum_alpha');
validateattributes(background_color, {'numeric'}, ...
    {'vector', 'numel', 3, 'real', 'finite', '>=', 0, '<=', 1}, ...
    mfilename, 'background_color');
validateattributes(initial_view_padding, {'numeric'}, ...
    {'scalar', 'real', 'finite', 'nonnegative'}, ...
    mfilename, 'initial_view_padding');
validateattributes(initial_zoom_factor, {'numeric'}, ...
    {'scalar', 'real', 'finite', 'positive'}, ...
    mfilename, 'initial_zoom_factor');

render_mode = lower(convertCharsToStrings(render_mode));
if ~ismember(render_mode, ["spheres", "markers"])
    error('CMEP:RenderMode', ...
        'render_mode must be "spheres" or "markers".');
end

%% Build the interactive 3D view
foreground_color = 1 - reshape(background_color, 1, 3);
figure_handle = figure( ...
    'Name', 'Localized CMEP Atomic Centers', ...
    'NumberTitle', 'off', ...
    'Color', background_color, ...
    'Position', figure_position);
axes_handle = axes(figure_handle, ...
    'Color', background_color, ...
    'XColor', foreground_color, ...
    'YColor', foreground_color, ...
    'ZColor', foreground_color, ...
    'GridColor', foreground_color, ...
    'MinorGridColor', foreground_color, ...
    'Projection', 'perspective', ...
    'Box', 'off');
hold(axes_handle, 'on');
grid(axes_handle, 'off');

color_min = min(atom_color_values);
color_max = max(atom_color_values);
if color_max > color_min
    normalized_color = (atom_color_values - color_min) ./ ...
        (color_max - color_min);
else
    normalized_color = ones(size(atom_color_values));
end
atom_alpha = minimum_alpha + normalized_color .* (1 - minimum_alpha);

if render_mode == "spheres"
    [vertices, faces, vertex_color_values, vertex_alpha] = ...
        buildSphereMesh(atom_positions_xyz, atom_color_values, atom_alpha, ...
        sphere_radius, sphere_resolution);
    atom_plot_handle = patch(axes_handle, ...
        'Vertices', vertices, ...
        'Faces', faces, ...
        'FaceVertexCData', vertex_color_values, ...
        'FaceVertexAlphaData', vertex_alpha, ...
        'FaceColor', 'interp', ...
        'FaceAlpha', 'interp', ...
        'AlphaDataMapping', 'none', ...
        'EdgeColor', 'none', ...
        'FaceLighting', 'gouraud', ...
        'AmbientStrength', 0.45, ...
        'DiffuseStrength', 0.70, ...
        'SpecularStrength', 0.20, ...
        'SpecularExponent', 15);
    camlight(axes_handle, 'headlight');
    camlight(axes_handle, 'right');
else
    atom_plot_handle = scatter3(axes_handle, ...
        atom_positions_xyz(:, 1), atom_positions_xyz(:, 2), ...
        atom_positions_xyz(:, 3), marker_size, atom_color_values, ...
        'filled', 'MarkerEdgeColor', 'none');
    atom_plot_handle.AlphaData = atom_alpha;
    atom_plot_handle.MarkerFaceAlpha = 'flat';
    atom_plot_handle.AlphaDataMapping = 'none';
end

if has_color_column
    colormap(axes_handle, resolveColormap(colormap_name, 256));
    if color_max > color_min
        clim(axes_handle, [color_min, color_max]);
    else
        clim(axes_handle, color_min + [-0.5, 0.5]);
    end
else
    if render_mode == "spheres"
        atom_plot_handle.FaceColor = fixed_color;
    else
        atom_plot_handle.CData = fixed_color;
        atom_plot_handle.MarkerFaceColor = fixed_color;
    end
end

coordinate_minimum = min(atom_positions_xyz, [], 1);
coordinate_maximum = max(atom_positions_xyz, [], 1);
coordinate_span = coordinate_maximum - coordinate_minimum;
reference_span = max(coordinate_span);
if reference_span <= 0
    reference_span = max(2 * sphere_radius, 1);
end
axis_padding = max(coordinate_span .* initial_view_padding, ...
    reference_span .* initial_view_padding);
if render_mode == "spheres"
    axis_padding = max(axis_padding, sphere_radius);
end
plot_minimum = coordinate_minimum - axis_padding;
plot_maximum = coordinate_maximum + axis_padding;

xlim(axes_handle, [plot_minimum(1), plot_maximum(1)]);
ylim(axes_handle, [plot_minimum(2), plot_maximum(2)]);
zlim(axes_handle, [plot_minimum(3), plot_maximum(3)]);
axis(axes_handle, 'equal');
axis(axes_handle, 'vis3d');
daspect(axes_handle, [1, 1, 1]);
view(axes_handle, camera_azimuth, camera_elevation);
camup(axes_handle, [0, 0, 1]);
camtarget(axes_handle, 0.5 .* (coordinate_minimum + coordinate_maximum));

xlabel(axes_handle, "x (" + length_unit + ")", 'Color', foreground_color);
ylabel(axes_handle, "y (" + length_unit + ")", 'Color', foreground_color);
zlabel(axes_handle, "z (" + length_unit + ")", 'Color', foreground_color);
title(axes_handle, sprintf('Localized atomic centers: %s atoms', ...
    formatInteger(size(atom_positions_xyz, 1))), ...
    'Color', foreground_color, 'FontWeight', 'normal');

if show_colorbar && has_color_column
    colorbar_handle = colorbar(axes_handle);
    colorbar_handle.Color = foreground_color;
    colorbar_handle.Label.String = strrep(char(color_column), '_', ' ');
    colorbar_handle.Label.Color = foreground_color;
end

drawnow;
camzoom(axes_handle, initial_zoom_factor);

try
    axtoolbar(axes_handle, {'rotate', 'pan', 'zoomin', 'zoomout', ...
        'restoreview'});
catch toolbar_error
    warning('CMEP:AxesToolbar', ...
        'Could not create the axes toolbar: %s', toolbar_error.message);
end
rotate3d(figure_handle, 'on');

fprintf('Localized 3D structure\n');
fprintf('  CSV: %s\n', csv_path);
fprintf('  displayed centers: %s\n', ...
    formatInteger(size(atom_positions_xyz, 1)));
fprintf('  excluded rows: %s\n', formatInteger(discarded_count));
fprintf('  mode: %s\n', render_mode);
if render_mode == "spheres"
    fprintf('  display sphere radius: %.6g %s\n', ...
        sphere_radius, length_unit);
end
fprintf('  likelihood range: %.6g to %.6g\n', color_min, color_max);

%% Local functions
function values = getNumericColumn(table_data, names, requested_name)
    column_index = find(strcmpi(names, requested_name), 1, 'first');
    if isempty(column_index)
        error('CMEP:MissingColumn', ...
            'CSV column "%s" was not found.', requested_name);
    end
    values = table_data{:, column_index};
    if ~isnumeric(values) || ~isvector(values)
        error('CMEP:NonNumericColumn', ...
            'CSV column "%s" must be numeric.', requested_name);
    end
    values = double(values(:));
end

function [vertices, faces, vertex_values, vertex_alpha] = buildSphereMesh( ...
        positions, color_values, atom_alpha, radius, resolution)
    [sphere_x, sphere_y, sphere_z] = sphere(resolution);
    template = surf2patch(sphere_x, sphere_y, sphere_z, 'triangles');
    template_vertices = single(template.vertices) .* single(radius);
    template_faces = uint32(template.faces);

    atom_count = size(positions, 1);
    vertices_per_atom = size(template_vertices, 1);
    faces_per_atom = size(template_faces, 1);

    vertices = repmat(template_vertices, atom_count, 1) + ...
        repelem(single(positions), vertices_per_atom, 1);
    face_offsets = repelem( ...
        uint32((0:atom_count - 1)' .* vertices_per_atom), ...
        faces_per_atom, 1);
    faces = repmat(template_faces, atom_count, 1) + face_offsets;
    vertex_values = repelem(single(color_values), vertices_per_atom, 1);
    vertex_alpha = repelem(single(atom_alpha), vertices_per_atom, 1);
end

function color_map = resolveColormap(name, count)
    name = lower(strtrim(string(name)));
    if name == "magma"
        color_map = magmaColormap(count);
        return;
    end
    try
        color_map = feval(char(name), count);
    catch map_error
        error('CMEP:Colormap', ...
            'Unknown MATLAB colormap "%s": %s', name, map_error.message);
    end
    if ~isnumeric(color_map) || size(color_map, 2) ~= 3
        error('CMEP:ColormapShape', ...
            'Colormap "%s" did not return an N-by-3 numeric array.', name);
    end
end

function color_map = magmaColormap(count)
    anchor_rgb = [ ...
          0,   0,   4;  16,  12,  48;  38,  18,  79; ...
         64,  15, 104;  92,  18, 110; 120,  28, 109; ...
        149,  38, 103; 177,  51,  91; 203,  70,  77; ...
        225,  94,  64; 242, 123,  55; 252, 157,  58; ...
        254, 194,  83; 252, 232, 131; 252, 253, 191] ./ 255;
    anchor_position = linspace(0, 1, size(anchor_rgb, 1));
    target_position = linspace(0, 1, count);
    color_map = interp1(anchor_position, anchor_rgb, target_position, ...
        'linear');
    color_map = min(max(color_map, 0), 1);
end

function text_value = formatInteger(value)
    text_value = sprintf('%.0f', value);
    for insertion = length(text_value) - 3:-3:1
        text_value = [text_value(1:insertion), ',', ...
            text_value(insertion + 1:end)];
    end
end
