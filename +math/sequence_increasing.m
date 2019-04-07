%
% superclass for all strictly monotonically increasing sequences
%
% author: Martin F. Schiffner
% date: 2019-03-29
% modified: 2019-04-02
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
        % cardinality (overload abs function)
        %------------------------------------------------------------------
        function N_members = abs( sequences )

            % initialize N_members with zeros
            N_members = zeros( size( sequences ) );

            % iterate sequences
            for index_object = 1:numel( sequences )
                N_members( index_object ) = numel( sequences( index_object ).members );
            end

        end % function N_members = abs( sequences )

        %------------------------------------------------------------------
        % unique values in array (overload unique function)
        %------------------------------------------------------------------
        function [ sequence_out, indices_unique_to_f, indices_f_to_unique ] = unique( sequences_in )

            %--------------------------------------------------------------
            % 1.) numbers of members and cumulative sum
            %--------------------------------------------------------------
            N_members = abs( sequences_in(:) );
            N_members_cs = [ 0; cumsum( N_members ) ];

            %--------------------------------------------------------------
            % 2.) create sequence of unique members
            %--------------------------------------------------------------
            % extract unique discrete frequencies
            [ members_unique, ia, ic ] = unique( [ sequences_in.members ] );
            N_members_unique = numel( members_unique );

            % create sequence of unique members
            sequence_out = math.sequence_increasing( members_unique );

            %--------------------------------------------------------------
            % 3.) map unique members to those in each sequence
            %--------------------------------------------------------------
            % object and frequency indices for each unique frequency
            indices_object = sum( ( repmat( ia, [ 1, numel( N_members_cs ) ] ) - repmat( N_members_cs(:)', [ N_members_unique, 1 ] ) ) > 0, 2 );
            indices_f = ia - N_members_cs( indices_object );

            % create structures with object and frequency indices for each unique frequency
            indices_unique_to_f( N_members_unique ).index_object = indices_object( N_members_unique );
            indices_unique_to_f( N_members_unique ).index_f = indices_f( N_members_unique );
            for index_f_unique = 1:(N_members_unique-1)
                indices_unique_to_f( index_f_unique ).index_object = indices_object( index_f_unique );
                indices_unique_to_f( index_f_unique ).index_f = indices_f( index_f_unique );
            end

            %--------------------------------------------------------------
            % 4.) map members in each sequence to the unique members
            %--------------------------------------------------------------
            indices_f_to_unique = cell( size( sequences_in ) );

            for index_set = 1:numel( sequences_in )

                index_start = N_members_cs( index_set ) + 1;
                index_stop = index_start + N_members( index_set ) - 1;

                indices_f_to_unique{ index_set } = ic( index_start:index_stop );
            end

        end % function [ sequence_out, indices_unique_to_f, indices_f_to_unique ] = unique( sequences_in )

    end % methods

end % classdef sequence_increasing