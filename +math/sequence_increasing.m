%
% superclass for all strictly monotonically increasing sequences
%
% author: Martin F. Schiffner
% date: 2019-03-29
% modified: 2020-02-03
%
classdef sequence_increasing

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        members ( :, 1 ) physical_values.physical_quantity

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = sequence_increasing( members )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % return default sequence
            if nargin == 0
                return;
            end

            % ensure cell array for members
            if ~iscell( members )
                members = { members };
            end

            %--------------------------------------------------------------
            % 2.) create strictly monotonically increasing sequences
            %--------------------------------------------------------------
            % repeat default sequence
            objects = repmat( objects, size( members ) );

            % iterate sequences
            for index_object = 1:numel( members )

                % ensure class physical_values.physical_quantity
                if ~isa( members{ index_object }, 'physical_values.physical_quantity' )
                    errorStruct.message = sprintf( 'members{ %d } must be physical_values.physical_quantity!', index_object );
                    errorStruct.identifier = 'sequence_increasing:NoPhysicalQuantity';
                    error( errorStruct );
                end

                % ensure column vector
                if ~iscolumn( members{ index_object } )
                    errorStruct.message = sprintf( 'members{ %d } must be a column vector!', index_object );
                    errorStruct.identifier = 'sequence_increasing:NoColumnVector';
                    error( errorStruct );
                end

                % ensure strictly monotonic increase
                if ~issorted( members{ index_object }, 'strictascend' )
                    errorStruct.message = sprintf( 'members{ %d } must be strictly monotonically increasing!', index_object );
                    errorStruct.identifier = 'sequence_increasing:NoStrictIncrease';
                    error( errorStruct );
                end

                % set independent properties
                objects( index_object ).members = members{ index_object };

            end % for index_object = 1:numel( members )

        end % function objects = sequence_increasing( members )

        %------------------------------------------------------------------
        % unique values in array (overload unique function)
        %------------------------------------------------------------------
        function [ sequence_out, indices_unique_to_local, indices_local_to_unique ] = unique( sequences_in )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class math.sequence_increasing
            if ~isa( sequences_in, 'math.sequence_increasing' )
                errorStruct.message = 'sequences_in must be math.sequence_increasing!';
                errorStruct.identifier = 'unique:NoIncreasingSequences';
                error( errorStruct );
            end

            % ensure equal subclasses of physical_values.physical_quantity
            auxiliary.mustBeEqualSubclasses( 'physical_values.physical_quantity', sequences_in.members );

            %--------------------------------------------------------------
            % 2.) numbers of members and cumulative sum
            %--------------------------------------------------------------
            N_members = abs( sequences_in( : ) );
            N_members_cs = [ 0; cumsum( N_members ) ];

            %--------------------------------------------------------------
            % 3.) create sequence of unique members
            %--------------------------------------------------------------
            % extract unique members in sorted order
            members_cell = { sequences_in.members };
            [ members_unique, ia, ic ] = unique( cat( 1, members_cell{ : } ) );
            N_members_unique = numel( members_unique );

            % compute deltas
% TODO: use auxiliary.isregular
            deltas = diff( members_unique );

            % check regularity
            if all( abs( deltas - deltas( 1 ) ) < 1e-10 * deltas( 1 ) )
                % create regular increasing sequence
                q_lb = round( members_unique( 1 ) / deltas( 1 ) );
                q_ub = round( members_unique( end ) / deltas( 1 ) );
                sequence_out = math.sequence_increasing_regular_quantized( q_lb, q_ub, deltas( 1 ) );
            else
                % create increasing sequence
                sequence_out = math.sequence_increasing( members_unique );
            end

            %--------------------------------------------------------------
            % 4.) map unique members to those in each sequence
            %--------------------------------------------------------------
            % object and member indices for each unique member
            indices_object = sum( ( repmat( ia, [ 1, numel( N_members_cs ) ] ) - repmat( N_members_cs( : )', [ N_members_unique, 1 ] ) ) > 0, 2 );
            indices_f = ia - N_members_cs( indices_object );

            % create structures with object and member indices for each unique member
            indices_unique_to_local( N_members_unique ).index_object = indices_object( N_members_unique );
            indices_unique_to_local( N_members_unique ).index_f = indices_f( N_members_unique );
            for index_f_unique = 1:( N_members_unique - 1 )
                indices_unique_to_local( index_f_unique ).index_object = indices_object( index_f_unique );
                indices_unique_to_local( index_f_unique ).index_f = indices_f( index_f_unique );
            end

            %--------------------------------------------------------------
            % 5.) map members in each sequence to the unique members
            %--------------------------------------------------------------
            indices_local_to_unique = cell( size( sequences_in ) );

            for index_set = 1:numel( sequences_in )

                index_start = N_members_cs( index_set ) + 1;
                index_stop = index_start + N_members( index_set ) - 1;

                indices_local_to_unique{ index_set } = ic( index_start:index_stop );
            end

        end % function [ sequence_out, indices_unique_to_local, indices_local_to_unique ] = unique( sequences_in )

        %------------------------------------------------------------------
        % subsample
        %------------------------------------------------------------------
        function sequences_out = subsample( sequences, indices )
% TODO: only a single indice -> check for regularity fails
            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class math.sequence_increasing
            if ~isa( sequences, 'math.sequence_increasing' )
                errorStruct.message = 'sequences must be math.sequence_increasing!';
                errorStruct.identifier = 'subsample:NoIncreasingSequences';
                error( errorStruct );
            end

            % ensure cell array for indices
            if ~iscell( indices )
                indices = { indices };
            end

            % multiple sequences / single indices
            if ~isscalar( sequences ) && isscalar( indices )
                indices = repmat( indices, size( sequences ) );
            end

            % single sequences / multiple indices
            if isscalar( sequences ) && ~isscalar( indices )
                sequences = repmat( sequences, size( indices ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( sequences, indices );

            %--------------------------------------------------------------
            % 2.) perform subsampling
            %--------------------------------------------------------------
            % extract cardinalities
            N_members = abs( sequences );

            % specify cell array for sequences_out
            sequences_out = cell( size( sequences ) );

            % iterate sequences
            for index_object = 1:numel( sequences )

                % ensure positive integers
                mustBeInteger( indices{ index_object } );
                mustBePositive( indices{ index_object } );

                % ensure that indices{ index_object } do not exceed N_members
                if any( indices{ index_object } > N_members( index_object ) )
                    errorStruct.message = sprintf( 'indices{ %d } must not exceed %d!', index_object, N_members( index_object ) );
                    errorStruct.identifier = 'subsample:InvalidIndices';
                    error( errorStruct );
                end

                % ensure strictly monotonic increase
                if ~issorted( indices{ index_object }, 'strictascend' )
                    errorStruct.message = sprintf( 'indices{ %d } must be strictly monotonically increasing!', index_object );
                    errorStruct.identifier = 'subsample:NoStrictIncrease';
                    error( errorStruct );
                end

                % subsample members
                sequences( index_object ).members = sequences( index_object ).members( indices{ index_object } );

                % compute deltas
                deltas = diff( sequences( index_object ).members );

                % check regularity
% TODO: check quantization!
                if all( abs( deltas( : ) - deltas( 1 ) ) < 1e-10 * deltas( 1 ) )
                    % convert to strictly monotonically increasing sequence with regular spacing
                    q_lb = round( sequences( index_object ).members( 1 ) / deltas( 1 ) );
                    q_ub = round( sequences( index_object ).members( end ) / deltas( 1 ) );
                    sequences_out{ index_object } = math.sequence_increasing_regular_quantized( q_lb, q_ub, deltas( 1 ) );
                else
                    % maintain current sequence
                    sequences_out{ index_object } = sequences( index_object );
                end

            end % for index_object = 1:numel( sequences )

            %
            if all( cellfun( @( x ) strcmp( class( x( : ) ), 'math.sequence_increasing' ), sequences_out ) ) || all( cellfun( @( x ) strcmp( class( x( : ) ), 'math.sequence_increasing_regular_quantized' ), sequences_out ) )
                sequences_out = reshape( [ sequences_out{ : } ], size( sequences ) );
            end

            % avoid cell array for single sequence
            if isscalar( sequences ) && iscell( sequences_out )
                sequences_out = sequences_out{ 1 };
            end

        end % function sequences_out = subsample( sequences, indices )

        %------------------------------------------------------------------
        % cut out subsequence
        %------------------------------------------------------------------
        function [ sequences, indicators ] = cut_out( sequences, lbs, ubs )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class math.sequence_increasing
            if ~isa( sequences, 'math.sequence_increasing' )
                errorStruct.message = 'sequences must be math.sequence_increasing!';
                errorStruct.identifier = 'cut_out:NoIncreasingSequences';
                error( errorStruct );
            end

            % ensure equal subclasses of physical_values.physical_quantity
            auxiliary.mustBeEqualSubclasses( 'physical_values.physical_quantity', sequences.members, lbs, ubs );

            % multiple sequences / single lbs
            if ~isscalar( sequences ) && isscalar( lbs )
                lbs = repmat( lbs, size( sequences ) );
            end

            % multiple sequences / single ubs
            if ~isscalar( sequences ) && isscalar( ubs )
                ubs = repmat( ubs, size( sequences ) );
            end

            % single sequences / multiple lbs
            if isscalar( sequences ) && ~isscalar( lbs )
                sequences = repmat( sequences, size( lbs ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( sequences, lbs, ubs );

            %--------------------------------------------------------------
            % 2.) perform cut out
            %--------------------------------------------------------------
            % specify cell array for indicators
            indicators = cell( size( sequences ) );

            % iterate sequences
            for index_object = 1:numel( sequences )

                % identify members to keep
                indicators{ index_object } = ( sequences( index_object ).members >= lbs( index_object ) ) & ( sequences( index_object ).members <= ubs( index_object ) );

                % cut out members
                sequences( index_object ).members = sequences( index_object ).members( indicators{ index_object } );

            end % for index_object = 1:numel( sequences )

            % avoid cell array for single sequence
            if isscalar( sequences )
                indicators = indicators{ 1 };
            end

        end % function [ sequences, indicators ] = cut_out( sequences, lbs, ubs )

        %------------------------------------------------------------------
        % remove last members
        %------------------------------------------------------------------
        function [ sequences, N_remove ] = remove_last( sequences, varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class math.sequence_increasing
            if ~isa( sequences, 'math.sequence_increasing' )
                errorStruct.message = 'sequences must be math.sequence_increasing!';
                errorStruct.identifier = 'remove_last:NoIncreasingSequences';
                error( errorStruct );
            end

            % ensure nonempty N_remove
            if nargin >= 2 && ~isempty( varargin{ 1 } )
                N_remove = varargin{ 1 };
            else
                N_remove = ones( size( sequences ) );
            end

            % ensure positive integers
            mustBePositive( N_remove );
            mustBeInteger( N_remove );

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( sequences, N_remove );

            % numbers of members in all sequences
            N_members = abs( sequences );

            % ensure that N_remove < N_members
            if any( N_remove >= N_members )
                errorStruct.message = 'N_remove must be smaller than N_members!';
                errorStruct.identifier = 'remove_last:InvalidNumber';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) remove last members
            %--------------------------------------------------------------
            % iterate sequences
            for index_object = 1:numel( sequences )

                % remove N_remove( index_object ) last members
                sequences( index_object ).members = sequences( index_object ).members( 1:( end - N_remove( index_object ) ) );

            end % for index_object = 1:numel( sequences )

        end % function [ sequences, N_remove ] = remove_last( sequences, varargin )

        %------------------------------------------------------------------
        % cardinality (overload abs function)
        %------------------------------------------------------------------
        function N_members = abs( sequences )

            % compute numbers of members
            N_members = reshape( cellfun( @numel, { sequences.members } ), size( sequences ) );

        end % function N_members = abs( sequences )

    end % methods

end % classdef sequence_increasing
