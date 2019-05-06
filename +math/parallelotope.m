%
% superclass for all d-parallelotopes
%
% author: Martin F. Schiffner
% date: 2019-03-21
% modified: 2019-04-22
%
classdef parallelotope

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        edge_lengths ( 1, : ) physical_values.length
        basis ( 1, : ) math.unit_vector

        % dependent properties
        volume ( 1, 1 ) physical_values.physical_quantity

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = parallelotope( edge_lengths, basis )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % return for no input arguments
            if nargin == 0
                return;
            end

            % ensure cell array for edge_lengths
            if ~iscell( edge_lengths )
                edge_lengths = { edge_lengths };
            end

            % ensure cell array for basis
            if ~iscell( basis )
                basis = { basis };
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( edge_lengths, basis );

            %--------------------------------------------------------------
            % 2.) create parallelotopes
            %--------------------------------------------------------------
            % repeat default parallelotope
            objects = repmat( objects, size( edge_lengths ) );

            % check and set independent properties
            for index_object = 1:numel( edge_lengths )

                % ensure class physical_values.length
                if ~isa( edge_lengths{ index_object }, 'physical_values.length' )
                    errorStruct.message = sprintf( 'edge_lengths{ %d } must be physical_values.length!', index_object );
                    errorStruct.identifier	= 'parallelotope:NoLengths';
                    error( errorStruct );
                end

                % ensure class math.unit_vector
                if ~isa( basis{ index_object }, 'math.unit_vector' )
                    errorStruct.message = sprintf( 'basis{ %d } must be math.unit_vector!', index_object );
                    errorStruct.identifier	= 'parallelotope:NoUnitVectors';
                    error( errorStruct );
                end

                % ensure equal number of dimensions and sizes
                auxiliary.mustBeEqualSize( edge_lengths{ index_object }, basis{ index_object } );

                % set independent properties
                objects( index_object ).edge_lengths = edge_lengths{ index_object };
                objects( index_object ).basis = basis{ index_object };

                % set dependent properties
                objects( index_object ).volume = compute_volume( objects( index_object ) );

                % ensure linearly independent unit vectors in basis
                if double( objects( index_object ).volume ) < eps
                    errorStruct.message = sprintf( 'basis{ %d } must consist of linearly independent math.unit_vector!', index_object );
                    errorStruct.identifier	= 'parallelotope:NoBasis';
                    error( errorStruct );
                end

            end % for index_object = 1:numel( edge_lengths )

        end % function objects = parallelotope( edge_lengths, basis )

        %------------------------------------------------------------------
        % volume
        %------------------------------------------------------------------
        function volumes = compute_volume( parallelotopes )

            % initialize cell array to enable diverse physical units
            volumes = cell( size( parallelotopes ) );

            % iterate parallelotopes
            for index_object = 1:numel( parallelotopes )

                % compute volume as absolute value of the determinant of the basis vectors
                volumes{ index_object } = abs( det( parallelotopes( index_object ).edge_lengths .* parallelotopes( index_object ).basis ) );

            end % for index_object = 1:numel( parallelotopes )

            % avoid cell array for single parallelotope
            if isscalar( parallelotopes )
                volumes = volumes{ 1 };
            end

        end % function volumes = compute_volume( parallelotopes )

    end % methods

end % classdef parallelotope
