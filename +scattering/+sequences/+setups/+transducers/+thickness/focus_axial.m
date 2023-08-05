function thickness = focus_axial( axial_focus_axis, element_width_axis, positions_rel_norm )
% lateral focusing at specified axial distances
% 
% axial_focus_axis = axial distances of the lateral foci
% element_width_axis = widths of orthotope shape
% positions_rel_norm = normalized relative positions of the grid points
%
% author: Martin F. Schiffner
% date: 2019-08-23
% modified: 2019-08-23

	%----------------------------------------------------------------------
	% 1.) check arguments
	%----------------------------------------------------------------------
	% ensure equal number of dimensions and sizes
	auxiliary.mustBeEqualSize( axial_focus_axis, element_width_axis );

	%----------------------------------------------------------------------
	% 2.) compute thickness profile
	%----------------------------------------------------------------------
    % find elements that are infinite
    indicator_inf = isinf( axial_focus_axis );

    % quick exit if all elements are infinite
    if all( indicator_inf )
        thickness = positions_rel_norm * 0;
        return;
    end

    % normalized relative positions of the grid points
    axial_focus_axis_norm = 2 * axial_focus_axis ./ element_width_axis;

    % maximum values
    max_vals = sqrt( 1 + axial_focus_axis_norm( ~indicator_inf ).^2 );

    % compute thickness
    thickness = sum( ( max_vals - sqrt( positions_rel_norm( :, ~indicator_inf ).^2 + axial_focus_axis_norm( ~indicator_inf ).^2 ) ) .* element_width_axis( ~indicator_inf ) / 2, 2 );

end % function thickness = focus_axial( axial_focus_axis, element_width_axis, positions_rel_norm )
