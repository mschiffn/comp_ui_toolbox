%
% superclass for all orthotopes of physical values
%
% author: Martin F. Schiffner
% date: 2019-02-11
% modified: 2019-04-02
%
classdef orthotope

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        intervals ( 1, : ) math.interval	% intervals of physical quantities

        % dependent properties
        N_dimensions ( 1, 1 ) { mustBeInteger, mustBeNonempty } = 0     % number of dimensions

    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = orthotope( varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % return if no argument
            if nargin == 0
                return;
            end

            % ensure equal subclasses of math.interval
            auxiliary.mustBeEqualSubclasses( 'math.interval', varargin{ : } );

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
        function objects_out = vertices( orthotopes )

            % initialize results
            objects_out = cell( size( orthotopes ) );

            % iterate orthotopes
            for index_object = 1:numel( orthotopes )

                % number of vertices
                N_dimensions_act = orthotopes( index_object ).N_dimensions;
                N_vertices_act = 2^N_dimensions_act;

                % initialize results
                objects_out{ index_object } = repmat( orthotopes( index_object ).intervals( 1 ).lb, [ N_vertices_act, N_dimensions_act ] );

                % extract vertices
                for index_vertex = 1:N_vertices_act
                    indices_bounds_act = dec2bin( index_vertex - 1, N_dimensions_act ) - '0' + 1;
                    for index_dimension = 1:N_dimensions_act
                        if indices_bounds_act( index_dimension ) == 1
                            objects_out{ index_object }( index_vertex, index_dimension ) = orthotopes( index_object ).intervals( index_dimension ).lb;
                        else
                            objects_out{ index_object }( index_vertex, index_dimension ) = orthotopes( index_object ).intervals( index_dimension ).ub;
                        end
                    end
                end

            end % for index_object = 1:numel( orthotopes )

            % avoid cell array for single orthotope
            if isscalar( orthotopes )
                objects_out = objects_out{ 1 };
            end

        end % function objects_out = vertices( orthotopes )

        %------------------------------------------------------------------
        % center
        %------------------------------------------------------------------
        function objects_out = center( orthotopes )

            % initialize results
            objects_out = cell( size( orthotopes ) );

            % iterate orthotopes
            for index_object = 1:numel( orthotopes )

                % compute interval centers
                objects_out{ index_object } = center( orthotopes( index_object ).intervals );

            end % for index_object = 1:numel( objects )

            % avoid cell array for single orthotope
            if isscalar( orthotopes )
                objects_out = objects_out{ 1 };
            end

        end % function objects_out = center( orthotopes )

        %------------------------------------------------------------------
        % discretize
        %------------------------------------------------------------------
        function objects_out = discretize( orthotopes, parameters )
            % TODO: various types of discretization (subtype of regular grid) / parameter objects

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class discretizations.parameters
            if ~isa( parameters, 'discretizations.parameters' )
                errorStruct.message     = 'parameters must be discretizations.parameters!';
                errorStruct.identifier	= 'discretize:NoParameters';
                error( errorStruct );
            end

            % multiple orthotopes / scalar parameters
            if ~isscalar( orthotopes ) && isscalar( parameters )
                parameters = repmat( parameters, size( orthotopes ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( orthotopes, parameters );

            %--------------------------------------------------------------
            % 2.) compute parameters for regular grids
            %--------------------------------------------------------------
            % initialize parameter cell arrays
            N_points_axis = cell( size( orthotopes ) );
            delta_axis = cell( size( orthotopes ) );
            offset_axis = cell( size( orthotopes ) );

            % iterate orthotopes
            for index_object = 1:numel( orthotopes )

                switch class( parameters( index_object ) )

                    case 'discretizations.parameters_number'

                        % number of grid points along each axis
                        N_points_axis{ index_object } = parameters( index_object ).values;

                        % ensure equal number of dimensions and sizes
                        auxiliary.mustBeEqualSize( orthotopes( index_object ).intervals, N_points_axis{ index_object } );

                        % distances between adjacent grid points along each axis
                        delta_axis{ index_object } = abs( orthotopes( index_object ).intervals ) ./ N_points_axis{ index_object };

                    case 'discretizations.parameters_distance'

                        % distances between adjacent grid points along each axis
                        delta_axis{ index_object } = parameters( index_object ).values;

                        % ensure equal number of dimensions and sizes
                        auxiliary.mustBeEqualSize( orthotopes( index_object ).intervals, delta_axis{ index_object } );

                        % number of grid points along each axis
                        N_points_axis{ index_object } = floor( abs( orthotopes( index_object ).intervals ) ./ delta_axis{ index_object } );

                    otherwise

                        errorStruct.message     = sprintf( 'Unknown class of parameters( %d )!', index_object );
                        errorStruct.identifier	= 'discretize:UnknownParameters';
                        error( errorStruct );

                end % switch class( parameters( index_object ) )

                % offset along each axis
                M_points_axis = ( N_points_axis{ index_object } - 1 ) / 2;
                offset_axis{ index_object } = center( orthotopes( index_object ) ) - M_points_axis .* delta_axis{ index_object };

            end % for index_object = 1:numel( orthotopes )

            %--------------------------------------------------------------
            % 3.) create orthogonal regular grids
            %--------------------------------------------------------------
            objects_out = discretizations.grid_regular_orthogonal( offset_axis, delta_axis, N_points_axis );

        end % function objects_out = discretize( orthotopes, parameters )

    end % methods

end % classdef orthotope
