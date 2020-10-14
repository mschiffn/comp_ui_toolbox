% superclass for all geometric orthotopes
%
% author: Martin F. Schiffner
% date: 2019-08-20
% modified: 2020-04-14
%
classdef orthotope < scattering.sequences.setups.geometry.shape & math.orthotope

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
                varargin = repmat( { math.interval( physical_values.meter( 0 ), physical_values.meter( 1 ) ) }, [ 2, 1 ] );
            end

% TODO: ensure class physical_values.length

            %--------------------------------------------------------------
            % 2.) create geometric orthotopes
            %--------------------------------------------------------------
            % constructors of superclasses
            objects@math.orthotope( varargin{ : } );
            objects@scattering.sequences.setups.geometry.shape( 1 );

        end % function objects = orthotope( varargin )

        %------------------------------------------------------------------
        % discretize
        %------------------------------------------------------------------
        function orthotopes_discrete = discretize( orthotopes, methods )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class scattering.sequences.setups.geometry.orthotope
            if ~isa( orthotopes, 'scattering.sequences.setups.geometry.orthotope' )
                errorStruct.message = 'orthotopes must be scattering.sequences.setups.geometry.orthotope!';
                errorStruct.identifier = 'discretize:NoOrthotopes';
                error( errorStruct );
            end

            % ensure class scattering.sequences.setups.discretizations.methods.method
            if ~isa( methods, 'scattering.sequences.setups.discretizations.methods.method' )
                errorStruct.message = 'methods must be scattering.sequences.setups.discretizations.methods.method!';
                errorStruct.identifier = 'discretize:NoSpatialDiscretizationMethod';
                error( errorStruct );
            end

            % multiple orthotopes / single methods
            if ~isscalar( orthotopes ) && isscalar( methods )
                methods = repmat( methods, size( orthotopes ) );
            end

            % single orthotopes / multiple methods
            if isscalar( orthotopes ) && ~isscalar( methods )
                orthotopes = repmat( orthotopes, size( methods ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( orthotopes, methods );

            %--------------------------------------------------------------
            % 2.) discretize orthotopes
            %--------------------------------------------------------------
            % check discretization method
            if isa( methods, 'scattering.sequences.setups.discretizations.methods.grid' )
% TODO: define grid subtype
                %----------------------------------------------------------
                % a) discretization based on (regular) grids
                %----------------------------------------------------------
                % specify cell arrays
                offset_axis = cell( size( orthotopes ) );
                delta_axis = cell( size( orthotopes ) );
                N_points_axis = cell( size( orthotopes ) );

                % check discretization subtype
% TODO: move functionality to scattering.sequences.setups.discretizations.methods.grid
                if isa( methods, 'scattering.sequences.setups.discretizations.methods.grid_numbers' )

                    %------------------------------------------------------
                    % i.) fixed number of grid points along each axis
                    %------------------------------------------------------
                    % iterate orthotopes
                    for index_object = 1:numel( orthotopes )

                        % number of grid points along each axis
                        N_points_axis{ index_object } = methods( index_object ).numbers';

                        % ensure equal number of dimensions and sizes
                        auxiliary.mustBeEqualSize( orthotopes( index_object ).intervals, N_points_axis{ index_object } );

                        % distances between adjacent grid points along each axis
                        delta_axis{ index_object } = abs( orthotopes( index_object ).intervals ) ./ N_points_axis{ index_object };

                    end % for index_object = 1:numel( orthotopes )

                elseif isa( methods, 'scattering.sequences.setups.discretizations.methods.grid_distances' )

                    %------------------------------------------------------
                    % ii.) fixed distances between adjacent grid points along each axis
                    %------------------------------------------------------
                    % iterate orthotopes
                    for index_object = 1:numel( orthotopes )

                        % distances between adjacent grid points along each axis
                        delta_axis{ index_object } = methods( index_object ).distances';

                        % ensure equal number of dimensions and sizes
                        auxiliary.mustBeEqualSize( orthotopes( index_object ).intervals, delta_axis{ index_object } );

                        % number of grid points along each axis
                        N_points_axis{ index_object } = floor( abs( orthotopes( index_object ).intervals ) ./ delta_axis{ index_object } );

                    end % for index_object = 1:numel( orthotopes )

                else

                    %------------------------------------------------------
                    % iii.) unknown discretization subtype
                    %------------------------------------------------------
                    errorStruct.message = sprintf( 'Unknown class of parameters( %d )!', index_object );
                    errorStruct.identifier = 'discretize:UnknownParameters';
                    error( errorStruct );

                end % if isa( methods, 'scattering.sequences.setups.discretizations.methods.grid_numbers' )

                % iterate orthotopes
                for index_object = 1:numel( orthotopes )

                    % offset along each axis
                    M_points_axis = ( N_points_axis{ index_object } - 1 ) / 2;
                    offset_axis{ index_object } = center( orthotopes( index_object ) ) - M_points_axis .* delta_axis{ index_object };

                end % for index_object = 1:numel( orthotopes )

                %----------------------------------------------------------
                % 3.) create orthogonal regular grids
                %----------------------------------------------------------
                grids = math.grid_regular_orthogonal( offset_axis, delta_axis, N_points_axis );
                intervals = { orthotopes.intervals };
                intervals = cat( 1, intervals{ : } );
                intervals = mat2cell( intervals, size( intervals , 1 ), ones( 1, size( intervals, 2 ) ) );
                orthotopes_discrete = scattering.sequences.setups.geometry.orthotope_grid( grids, intervals{ : } );

            else

                %----------------------------------------------------------
                % b) unknown discretization method
                %----------------------------------------------------------
                errorStruct.message = 'Class of methods is unknown!';
                errorStruct.identifier = 'discretize:UnknownMethodClass';
                error( errorStruct );

            end % if isa( methods, 'scattering.sequences.setups.discretizations.methods.grid' )

        end % function orthotopes_discrete = discretize( orthotopes, methods )

    end % methods

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods (protected and hidden)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods (Access = protected, Hidden)

        %------------------------------------------------------------------
        % check membership (scalar)
        %------------------------------------------------------------------
        function tf = iselement_scalar( orthotope, positions )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % calling function ensures class scattering.sequences.setups.geometry.shape (scalar) for orthotope
            % calling function ensures matching subclasses of class physical_values.length for positions
            % calling function ensures matching number of dimensions for positions

            %--------------------------------------------------------------
            % 2.) check membership (scalar)
            %--------------------------------------------------------------
            % initialize tf w/ false
            tf = false( size( positions ) );

            % iterate dimensions
            for index_dim = 1:size( positions, 2 )

                tf( :, index_dim ) = positions( :, index_dim ) >= orthotope.intervals( index_dim ).lb;
                tf( :, index_dim ) = tf( :, index_dim ) & ( positions( :, index_dim ) <= orthotope.intervals( index_dim ).ub );

            end % for index_dim = 1:size( positions, 2 )

            % test whether all inequalities are valid
            tf = all( tf, 2 );

        end % function tf = iselement_scalar( orthotope, positions )

        %------------------------------------------------------------------
        % draw (scalar)
        %------------------------------------------------------------------
        function draw_scalar( orthotope )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % calling function ensures class scattering.sequences.setups.geometry.shape (scalar) for orthotope

            %--------------------------------------------------------------
            % 2.) draw (scalar)
            %--------------------------------------------------------------
            % lateral bottom
            line( [ orthotope.intervals( 1 ).lb, orthotope.intervals( 1 ).ub ] * 1e3, orthotope.intervals( 3 ).lb * ones( 1, 2 ) * 1e3 );
            % lateral top
            line( [ orthotope.intervals( 1 ).lb, orthotope.intervals( 1 ).ub ] * 1e3, orthotope.intervals( 3 ).ub * ones( 1, 2 ) * 1e3 );
            % axial left
            line( orthotope.intervals( 1 ).lb * ones( 1, 2 ) * 1e3, [ orthotope.intervals( 3 ).lb, orthotope.intervals( 3 ).ub ] * 1e3 );
            % axial right
            line( orthotope.intervals( 1 ).ub * ones( 1, 2 ) * 1e3, [ orthotope.intervals( 3 ).lb, orthotope.intervals( 3 ).ub ] * 1e3 );

        end % function draw_scalar( orthotope )

	end % methods (Access = protected, Hidden)

end % classdef orthotope < scattering.sequences.setups.geometry.shape & math.orthotope
