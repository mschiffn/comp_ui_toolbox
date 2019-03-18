%
% superclass for all orthotopes of physical values
%
% author: Martin F. Schiffner
% date: 2019-02-11
% modified: 2019-03-18
%
classdef orthotope

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        intervals ( 1, : ) physical_values.interval                     % intervals of physical values

        % dependent properties
        N_dimensions ( 1, 1 ) { mustBeInteger, mustBeNonempty } = 0     % number of dimensions
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = orthotope( varargin )

            % return if no argument
            if nargin == 0
                return;
            end

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure equal subclasses of physical_values.interval
            auxiliary.mustBeEqualSubclasses( 'physical_values.interval', varargin{ : } );

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( varargin{ : } );

            %--------------------------------------------------------------
            % 2.) create orthotopes
            %--------------------------------------------------------------
            objects = repmat( objects, size( varargin{ 1 } ) );

            % set independent and dependent properties
            for index_object = 1:numel( varargin{ 1 } )

                %----------------------------------------------------------
                % a) set independent properties
                %----------------------------------------------------------                
                % initialize intervals
                objects( index_object ).intervals = repmat( varargin{ 1 }( index_object ), [ 1, nargin ] );

                % set intervals
                for index_argument = 2:nargin
                    objects( index_object ).intervals( index_argument ) = varargin{ index_argument }( index_object );
                end

                %----------------------------------------------------------
                % b) set dependent properties
                %----------------------------------------------------------
                objects( index_object ).N_dimensions = nargin;

            end % for index_object = 1:numel( varargin{ 1 } )

        end % function objects = orthotope( varargin )

        %------------------------------------------------------------------
        % vertices
        %------------------------------------------------------------------
        function objects_out = vertices( objects_in )

            % initialize results
            objects_out = cell( size( objects_in ) );

            % iterate orthotopes
            for index_object = 1:numel( objects_in )

                % number of vertices
                N_dimensions_act = objects_in( index_object ).N_dimensions;
                N_vertices_act = 2^N_dimensions_act;

                % initialize results
                objects_out{ index_object } = repmat( objects_in( index_object ).intervals( 1 ).lb, [ N_vertices_act, N_dimensions_act ] );

                % extract vertices
                for index_vertex = 1:N_vertices_act
                    indices_bounds_act = dec2bin( index_vertex - 1, N_dimensions_act ) - '0' + 1;
                    for index_dimension = 1:N_dimensions_act
                        if indices_bounds_act( index_dimension ) == 1
                            objects_out{ index_object }( index_vertex, index_dimension ) = objects_in( index_object ).intervals( index_dimension ).lb;
                        else
                            objects_out{ index_object }( index_vertex, index_dimension ) = objects_in( index_object ).intervals( index_dimension ).ub;
                        end
                    end
                end

            end % for index_object = 1:numel( objects_in )

            if numel( objects_in ) == 1
                objects_out = objects_out{1};
            end

        end % function objects_out = vertices( objects_in )

        %------------------------------------------------------------------
        % center
        %------------------------------------------------------------------
        function objects_out = center( objects_in )

            % initialize results
            objects_out = cell( size( objects_in ) );

            % iterate orthotopes
            for index_object = 1:numel( objects_in )

                % compute interval centers
                objects_out{ index_object } = center( objects_in( index_object ).intervals );

            end % for index_object = 1:numel( objects )

            % do not return cell array for single object
            if numel( objects_in ) == 1
                objects_out = objects_out{1};
            end

        end % function objects_out = center( objects_in )

        %------------------------------------------------------------------
        % discretize
        %------------------------------------------------------------------
        function objects_out = discretize( orthotopes, delta_axis )

            % TODO: various types of discretization / parameter objects
            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure cell array
            if ~iscell( delta_axis )
                delta_axis = { delta_axis };
            end

            % multiple orthotopes / single delta_axis
            if ~isscalar( orthotopes ) && isscalar( delta_axis )
                delta_axis = repmat( delta_axis, size( orthotopes ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( orthotopes, delta_axis );

            %--------------------------------------------------------------
            % 2.) compute parameters for regular grids
            %--------------------------------------------------------------
            % initialize parameter cell arrays
            N_points_axis = cell( size( orthotopes ) );
            grid_offset_axis = cell( size( orthotopes ) );

            for index_object = 1:numel( orthotopes )

                % ensure equal number of dimensions and sizes
                auxiliary.mustBeEqualSize( orthotopes( index_object ).intervals, delta_axis{ index_object } );

                % ensure positive real numbers
                mustBeReal( delta_axis{ index_object } );
                mustBePositive( delta_axis{ index_object } );

                % number of grid points along each axis
                % TODO: check rounding errors
                intervals_act = orthotopes( index_object ).intervals;
                N_points_axis{ index_object } = floor( double( abs( intervals_act ) ) ./ delta_axis{ index_object } );

                % offset along each axis
                M_points_axis = ( N_points_axis{ index_object } - 1 ) / 2;
                grid_offset_axis{ index_object } = double( center( orthotopes( index_object ) ) ) - M_points_axis .* delta_axis{ index_object };

            end % for index_object = 1:numel( orthotopes )

            %--------------------------------------------------------------
            % 3.) create regular grids
            %--------------------------------------------------------------
            objects_out = discretizations.grid( N_points_axis, delta_axis, grid_offset_axis );

        end % function objects_out = discretize( orthotopes, delta_axis )

    end % methods

end % classdef orthotope
