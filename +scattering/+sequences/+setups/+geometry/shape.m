% abstract superclass for all geometric shapes
%
% author: Martin F. Schiffner
% date: 2019-08-20
% modified: 2020-04-14
%
classdef (Abstract) shape

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = shape( size )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure row vector for size
            if ~isrow( size )
                errorStruct.message = 'size must be a row vector!';
                errorStruct.identifier = 'shape:NoRowVector';
                error( errorStruct );
            end

            % ensure nonempty positive integers
            mustBePositive( size );
            mustBeInteger( size );
            mustBeNonempty( size );

            %--------------------------------------------------------------
            % 2.) create geometric shapes
            %--------------------------------------------------------------
            % repeat geometric shapes
            objects = repmat( objects, size );

        end % function objects = shape( size )

        %------------------------------------------------------------------
        % check membership
        %------------------------------------------------------------------
        function tf = iselement( shapes, positions )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class scattering.sequences.setups.geometry.shape
            if ~isa( shapes, 'scattering.sequences.setups.geometry.shape' )
                errorStruct.message = 'shapes must be scattering.sequences.setups.geometry.shape!';
                errorStruct.identifier = 'iselement:NoCylinders';
                error( errorStruct );
            end

            % ensure cell array for positions
            if ~iscell( positions )
                positions = { positions };
            end

            % ensure equal number of dimensions and sizes
            [ shapes, positions ] = auxiliary.ensureEqualSize( shapes, positions );

            %--------------------------------------------------------------
            % 2.) check membership
            %--------------------------------------------------------------
            % specify cell array for tf
            tf = cell( size( shapes ) );

            % iterate shapes
            for index_shape = 1:numel( shapes )

                %----------------------------------------------------------
                % a) check arguments
                %----------------------------------------------------------
                % ensure equal subclasses of class physical_values.length
                auxiliary.mustBeEqualSubclasses( 'physical_values.length', shapes( index_shape ).center, positions{ index_shape } );

                % ensure equal numbers of dimensions
                if size( positions{ index_shape }, 2 ) ~= shapes( index_shape ).N_dimensions
                    errorStruct.message = sprintf( 'Dimensions of positions{ %d } and shapes( %d ) must match!', index_shape, index_shape );
                    errorStruct.identifier = 'iselement:DimensionMismatch';
                    error( errorStruct );
                end

                %----------------------------------------------------------
                % b) check membership (scalar)
                %----------------------------------------------------------
                tf{ index_shape } = iselement_scalar( shapes( index_shape ), positions{ index_shape } );

            end % for index_shape = 1:numel( shapes )

            % avoid cell array for single shapes
            if isscalar( shapes )
                tf = tf{ 1 };
            end

        end % function tf = iselement( shapes, positions )

        %------------------------------------------------------------------
        % draw
        %------------------------------------------------------------------
        function draw( shapes )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class scattering.sequences.setups.geometry.shape
            if ~isa( shapes, 'scattering.sequences.setups.geometry.shape' )
                errorStruct.message = 'shapes must be scattering.sequences.setups.geometry.shape!';
                errorStruct.identifier = 'draw:NoShapes';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) draw
            %--------------------------------------------------------------
            for index_shape = 1:numel( shapes )

                % draw (scalar)
                draw_scalar( shapes( index_shape ) );

            end % for index_shape = 1:numel( shapes )

        end % function draw( shapes )

    end

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods (abstract)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods (Abstract)

        %------------------------------------------------------------------
        % center
        %------------------------------------------------------------------
        objects_out = center( shapes )

        %------------------------------------------------------------------
        % move
        %------------------------------------------------------------------
        shapes = move( shapes, centers )

        %------------------------------------------------------------------
        % discretize
        %------------------------------------------------------------------
        objects_out = discretize( shapes, options )

	end % methods (Abstract)

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods (abstract, protected, and hidden)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods (Abstract, Access = protected, Hidden)

        %------------------------------------------------------------------
        % check membership (scalar)
        %------------------------------------------------------------------
        tf = iselement_scalar( shape, positions );

        %------------------------------------------------------------------
        % draw (scalar)
        %------------------------------------------------------------------
        draw_scalar( shape );

	end % methods (Abstract, Access = protected, Hidden)

end % classdef (Abstract) shape
