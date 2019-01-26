%
% superclass for all fields of view with orthotope shape
%
% author: Martin F. Schiffner
% date: 2018-01-23
% modified: 2019-01-17
%
classdef orthotope < fields_of_view.field_of_view

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent geometric properties
        size_axis ( 1, : ) double { mustBeReal, mustBePositive }	% length of FOV (m)
        offset_axis ( 1, : ) double { mustBeReal }                  % offset along axis (m)

        % dependent discretization properties
        grid ( 1, 1 ) grids.grid                                    % regular grid
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function obj = orthotope( size_axis, offset_axis, varargin )

            % constructor of superclass
            obj@fields_of_view.field_of_view( numel( size_axis ) );

            % set independent properties
            % TODO:  % check number of dimensions
            obj.size_axis	= size_axis;
            obj.offset_axis	= offset_axis;
        end

        %------------------------------------------------------------------
        % discretize orthotope
        %------------------------------------------------------------------
        function obj = discretize( obj, delta_axis )

            % ensure positive real numbers
            mustBeReal( delta_axis );
            mustBePositive( delta_axis );
            % assertion: delta_axis are real-valued and positive

            % create regular grid
            N_points_axis = floor( obj.size_axis ./ delta_axis );
            M_points_axis = ( N_points_axis - 1 ) / 2;

            % positions of grid points
            grid_offset_axis = obj.offset_axis + 0.5 * obj.size_axis - M_points_axis .* delta_axis;
            obj.grid = grids.grid( N_points_axis, delta_axis, grid_offset_axis );
        end
    end

end % classdef orthotope
