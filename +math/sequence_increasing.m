%
% superclass for all strictly monotonically increasing sequences
%
% author: Martin F. Schiffner
% date: 2019-03-29
% modified: 2019-05-08
%
classdef sequence_increasing

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        members ( 1, : ) physical_values.physical_quantity

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
            % 1.) numbers of members and cumulative sum
            %--------------------------------------------------------------
            N_members = abs( sequences_in( : ) );
            N_members_cs = [ 0; cumsum( N_members ) ];

            %--------------------------------------------------------------
            % 2.) create sequence of unique members
            %--------------------------------------------------------------
            % extract unique members
            [ members_unique, ia, ic ] = unique( [ sequences_in.members ] );
            N_members_unique = numel( members_unique );

            % create sequence of unique members
% TODO: create regular sequence if possible; use diff function?
            sequence_out = math.sequence_increasing( members_unique );

            %--------------------------------------------------------------
            % 3.) map unique members to those in each sequence
            %--------------------------------------------------------------
            % object and member indices for each unique member
            indices_object = sum( ( repmat( ia, [ 1, numel( N_members_cs ) ] ) - repmat( N_members_cs(:)', [ N_members_unique, 1 ] ) ) > 0, 2 );
            indices_f = ia - N_members_cs( indices_object );

            % create structures with object and member indices for each unique member
            indices_unique_to_local( N_members_unique ).index_object = indices_object( N_members_unique );
            indices_unique_to_local( N_members_unique ).index_f = indices_f( N_members_unique );
            for index_f_unique = 1:( N_members_unique - 1 )
                indices_unique_to_local( index_f_unique ).index_object = indices_object( index_f_unique );
                indices_unique_to_local( index_f_unique ).index_f = indices_f( index_f_unique );
            end

            %--------------------------------------------------------------
            % 4.) map members in each sequence to the unique members
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
        function sequences_out = subsample( sequences_in, indices_axes )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure cell array for indices_axes
            if ~iscell( indices_axes )
                indices_axes = { indices_axes };
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( sequences_in, indices_axes );

            %--------------------------------------------------------------
            % 2.) perform subsampling
            %--------------------------------------------------------------
            % extract cardinalities
            N_members = abs( sequences_in );

            % specify cell array for members_sub
            members_sub = cell( size( sequences_in ) );

            % iterate sequences
            for index_object = 1:numel( sequences_in )

                % ensure positive integers
                mustBeInteger( indices_axes{ index_object } );
                mustBePositive( indices_axes{ index_object } );

                % ensure that indices_axes{ index_object } do not exceed N_members
                if any( indices_axes{ index_object } > N_members( index_object ) )
                    errorStruct.message = sprintf( 'indices_axes{ %d } must not exceed %d!', index_object, N_members( index_object ) );
                    errorStruct.identifier = 'subsample:InvalidIndices';
                    error( errorStruct );
                end

                % subsample members
                members_sub{ index_object } = sequences_in( index_object ).members( indices_axes{ index_object } );

            end % for index_object = 1:numel( sequences_in )

            %--------------------------------------------------------------
            % 3.) create sequences
            %--------------------------------------------------------------
            sequences_out = math.sequence_increasing( members_sub );

        end % function sequences_out = subsample( sequences_in, indices_axes )

        %------------------------------------------------------------------
        % cardinality (overload abs function)
        %------------------------------------------------------------------
        function N_members = abs( sequences )

            % compute numbers of members
            N_members = reshape( cellfun( @numel, { sequences.members } ), size( sequences ) );

        end % function N_members = abs( sequences )

    end % methods

end % classdef sequence_increasing
